import time
import json
import os
import pickle
import hashlib
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from google.oauth2 import service_account
import requests
import io
import numpy as np
import cv2
import threading
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('background_processor.log'),
        logging.StreamHandler()
    ]
)

class PhotoProcessor:
    def __init__(self):
        # Google Drive API Setup
        self.SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
        self.SERVICE_ACCOUNT_FILE = 'credentials.json'
        self.DRIVE_FOLDER_ID = '19f3MqbHayvuXJExVM45q99ExgZMyIyTG'  # Replace with your folder ID
        
        # Cache file paths
        self.CACHE_DIR = 'cache'
        self.ENCODINGS_CACHE_FILE = os.path.join(self.CACHE_DIR, 'face_encodings.pkl')
        self.METADATA_CACHE_FILE = os.path.join(self.CACHE_DIR, 'file_metadata.json')
        
        # Initialize Google Drive service
        self.service = self._init_google_drive()
        
        # Cache storage
        self.face_encodings = {}
        self.file_metadata = {}
        
        # Load existing cache
        self._load_cache()
        
        # Background processing settings
        self.processing_interval = 300  # 5 minutes
        self.is_running = False
        self.processing_thread = None

    def _init_google_drive(self):
        """Initialize Google Drive API service."""
        try:
            creds = service_account.Credentials.from_service_account_file(
                self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES)
            service = build('drive', 'v3', credentials=creds)
            logging.info("✅ Google Drive API service initialized successfully")
            return service
        except Exception as e:
            logging.error(f"❌ Failed to initialize Google Drive service: {e}")
            raise

    def _load_cache(self):
        """Load existing face encodings and metadata from cache."""
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)
            logging.info("Created cache directory")

        # Load face encodings
        if os.path.exists(self.ENCODINGS_CACHE_FILE):
            try:
                with open(self.ENCODINGS_CACHE_FILE, 'rb') as f:
                    self.face_encodings = pickle.load(f)
                logging.info(f"Loaded {len(self.face_encodings)} face encodings from cache")
            except Exception as e:
                logging.error(f"Failed to load face encodings cache: {e}")
                self.face_encodings = {}

        # Load file metadata
        if os.path.exists(self.METADATA_CACHE_FILE):
            try:
                with open(self.METADATA_CACHE_FILE, 'r') as f:
                    self.file_metadata = json.load(f)
                logging.info(f"Loaded metadata for {len(self.file_metadata)} files")
            except Exception as e:
                logging.error(f"Failed to load metadata cache: {e}")
                self.file_metadata = {}

    def _save_cache(self):
        """Save face encodings and metadata to cache."""
        try:
            # Save face encodings
            with open(self.ENCODINGS_CACHE_FILE, 'wb') as f:
                pickle.dump(self.face_encodings, f)
            
            # Save metadata
            with open(self.METADATA_CACHE_FILE, 'w') as f:
                json.dump(self.file_metadata, f, indent=2)
            
            logging.info("Cache saved successfully")
        except Exception as e:
            logging.error(f"Failed to save cache: {e}")

    def _extract_face_features_opencv(self, image_bytes):
        """Extract face features using pure OpenCV (no DeepFace/TensorFlow)."""
        try:
            # Convert bytes to numpy array
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Load face cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Extract face region
                x, y, w, h = faces[0]
                face_region = gray[y:y+h, x:x+w]
                
                # Resize to standard size for comparison
                face_region = cv2.resize(face_region, (128, 128))
                
                # Create a simple feature vector (histogram + edge features)
                # This is a simplified approach - in production you'd want more sophisticated features
                hist_features = cv2.calcHist([face_region], [0], None, [32], [0, 256]).flatten()
                
                # Normalize histogram
                hist_features = hist_features / (np.sum(hist_features) + 1e-8)
                
                # Add edge features
                edges = cv2.Canny(face_region, 50, 150)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                
                # Combine features
                features = np.concatenate([hist_features, [edge_density]])
                
                return features
            
            return None
        except Exception as e:
            logging.warning(f"Failed to extract face features: {e}")
            return None

    def _download_image(self, file_id):
        """Download image from Google Drive."""
        try:
            request_uri = self.service.files().get_media(fileId=file_id).uri
            response = requests.get(request_uri, headers={'Authorization': f'Bearer {self.service._credentials.token}'})
            response.raise_for_status()
            return response.content
        except Exception as e:
            logging.error(f"Failed to download image {file_id}: {e}")
            return None

    def _get_drive_files(self):
        """Get list of files from Google Drive folder."""
        try:
            results = self.service.files().list(
                q=f"'{self.DRIVE_FOLDER_ID}' in parents and mimeType contains 'image/'",
                fields="nextPageToken, files(id, name, modifiedTime, size)"
            ).execute()
            
            files = {}
            for item in results.get('files', []):
                files[item['id']] = {
                    'name': item['name'],
                    'modified_time': item['modifiedTime'],
                    'size': item.get('size', '0')
                }
            
            return files
        except Exception as e:
            logging.error(f"Failed to get Drive files: {e}")
            return {}

    def _process_new_files(self, drive_files):
        """Process only new or modified files."""
        new_files = []
        modified_files = []
        
        for file_id, drive_info in drive_files.items():
            if file_id not in self.file_metadata:
                # New file
                new_files.append(file_id)
                logging.info(f"New file detected: {drive_info['name']}")
            else:
                # Check if file was modified
                cached_info = self.file_metadata[file_id]
                if (drive_info['modified_time'] != cached_info['modified_time'] or 
                    drive_info['size'] != cached_info['size']):
                    modified_files.append(file_id)
                    logging.info(f"File modified: {drive_info['name']}")

        return new_files, modified_files

    def _process_files(self, file_ids, reason="processing"):
        """Process files and extract face features."""
        processed_count = 0
        
        for file_id in file_ids:
            try:
                drive_info = self._get_drive_files().get(file_id, {})
                if not drive_info:
                    continue
                
                logging.info(f"{reason.capitalize()} file: {drive_info['name']}")
                
                # Download image
                image_content = self._download_image(file_id)
                if not image_content:
                    continue
                
                # Extract face features using OpenCV
                face_features = self._extract_face_features_opencv(image_content)
                if face_features is not None:
                    # Store features and metadata
                    self.face_encodings[file_id] = face_features
                    self.file_metadata[file_id] = drive_info
                    processed_count += 1
                    logging.info(f"Successfully processed: {drive_info['name']}")
                else:
                    logging.warning(f"No face detected in: {drive_info['name']}")
                    
            except Exception as e:
                logging.error(f"Error processing file {file_id}: {e}")
                continue
        
        return processed_count

    def process_cycle(self):
        """Main processing cycle - check for new/modified files and process them."""
        try:
            logging.info("Starting processing cycle...")
            
            # Get current files from Drive
            drive_files = self._get_drive_files()
            if not drive_files:
                logging.warning("No files found in Drive folder")
                return
            
            # Identify new and modified files
            new_files, modified_files = self._process_new_files(drive_files)
            
            if not new_files and not modified_files:
                logging.info("No new or modified files detected - skipping processing")
                return
            
            # Process new files
            if new_files:
                processed = self._process_files(new_files, "processing new")
                logging.info(f"Processed {processed} new files")
            
            # Process modified files
            if modified_files:
                processed = self._process_files(modified_files, "reprocessing modified")
                logging.info(f"Processed {processed} modified files")
            
            # Save updated cache
            self._save_cache()
            
            logging.info(f"Processing cycle completed. Total encodings: {len(self.face_encodings)}")
            
        except Exception as e:
            logging.error(f"Error in processing cycle: {e}")

    def start_background_processing(self):
        """Start background processing thread."""
        if self.is_running:
            logging.warning("Background processing is already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.processing_thread.start()
        logging.info("Background processing started")

    def stop_background_processing(self):
        """Stop background processing thread."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        logging.info("Background processing stopped")

    def _background_worker(self):
        """Background worker thread."""
        while self.is_running:
            try:
                self.process_cycle()
            except Exception as e:
                logging.error(f"Background worker error: {e}")
            
            # Wait for next cycle
            time.sleep(self.processing_interval)

    def get_face_encodings(self):
        """Get current face encodings for the web app."""
        return self.face_encodings.copy()

    def get_file_metadata(self):
        """Get current file metadata for the web app."""
        return self.file_metadata.copy()

    def initial_processing(self):
        """Perform initial processing of all files (first run)."""
        logging.info("Starting initial processing of all files...")
        
        drive_files = self._get_drive_files()
        if not drive_files:
            logging.warning("No files found for initial processing")
            return
        
        # Process all files
        all_file_ids = list(drive_files.keys())
        processed = self._process_files(all_file_ids, "initial processing")
        
        # Save cache
        self._save_cache()
        
        logging.info(f"Initial processing completed. Processed {processed} files")

def main():
    """Main function to run the background processor."""
    try:
        processor = PhotoProcessor()
        
        # Perform initial processing if no cache exists
        if not processor.face_encodings:
            processor.initial_processing()
        
        # Start background processing
        processor.start_background_processing()
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Shutting down...")
            processor.stop_background_processing()
            
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()