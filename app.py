from flask import Flask, render_template, request, jsonify
import base64
import json
import os
import io
import numpy as np
from datetime import datetime
import requests
from googleapiclient.discovery import build
from google.oauth2 import service_account
from deepface import DeepFace
import cv2
import threading
import time
import pickle

app = Flask(__name__)

# Simple configuration
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'credentials.json'
DRIVE_FOLDER_ID = '11NcB_TBuSNjZQahKRuXMGqfWHvkK_7G-'

# Cache file paths
CACHE_DIR = 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)
FACE_ENCODINGS_CACHE_FILE = os.path.join(CACHE_DIR, 'face_encodings_cache.pkl')
FILE_METADATA_CACHE_FILE = os.path.join(CACHE_DIR, 'file_metadata_cache.json')

# Global variables
face_encodings = {}
file_metadata = {}
cache_last_updated = None

def load_cache_from_disk():
    """Load cached data from disk"""
    global face_encodings, file_metadata, cache_last_updated
    
    try:
        # Load face encodings (using pickle for numpy arrays)
        if os.path.exists(FACE_ENCODINGS_CACHE_FILE):
            with open(FACE_ENCODINGS_CACHE_FILE, 'rb') as f:
                face_encodings = pickle.load(f)
            print(f"✅ Loaded {len(face_encodings)} face encodings from cache")
        
        # Load file metadata (using JSON for simple data)
        if os.path.exists(FILE_METADATA_CACHE_FILE):
            with open(FILE_METADATA_CACHE_FILE, 'r') as f:
                file_metadata = json.load(f)
            print(f"✅ Loaded {len(file_metadata)} file metadata from cache")
        
        # Load cache timestamp
        if os.path.exists(os.path.join(CACHE_DIR, 'cache_timestamp.txt')):
            with open(os.path.join(CACHE_DIR, 'cache_timestamp.txt'), 'r') as f:
                cache_last_updated = f.read().strip()
            print(f"✅ Cache last updated: {cache_last_updated}")
            
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        face_encodings = {}
        file_metadata = {}

def save_cache_to_disk():
    """Save cached data to disk"""
    global face_encodings, file_metadata, cache_last_updated
    
    try:
        # Save face encodings
        with open(FACE_ENCODINGS_CACHE_FILE, 'wb') as f:
            pickle.dump(face_encodings, f)
        
        # Save file metadata
        with open(FILE_METADATA_CACHE_FILE, 'w') as f:
            json.dump(file_metadata, f, indent=2)
        
        # Save timestamp
        cache_last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(os.path.join(CACHE_DIR, 'cache_timestamp.txt'), 'w') as f:
            f.write(cache_last_updated)
        
        print(f"✅ Cache saved successfully at {cache_last_updated}")
        
    except Exception as e:
        print(f"❌ Error saving cache: {e}")

def init_google_drive():
    """Initialize Google Drive API"""
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)
        print("✅ Google Drive API initialized successfully")
        return service
    except Exception as e:
        print(f"❌ Failed to initialize Google Drive: {e}")
        return None

def extract_face_encoding(image_data):
    """Extract face encoding from selfie using DeepFace"""
    try:
        print("🔍 Starting face encoding extraction...")
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("❌ Failed to decode image")
            return None
        
        print(f"✅ Image decoded successfully: {img.shape}")
        
        # Convert BGR to RGB (DeepFace expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Save debug image to see what's being processed
        cv2.imwrite('debug_selfie.jpg', img)
        print("💾 Debug image saved as 'debug_selfie.jpg'")
        
        # Extract face encoding using DeepFace
        print("🔍 Using DeepFace to extract face encoding...")
        face_encoding = DeepFace.represent(
            img_path=img_rgb,
            model_name="Facenet",
            detector_backend="opencv"
        )
        
        if face_encoding:
            print(f"✅ Face encoding extracted successfully: {len(face_encoding)} features")
            return face_encoding[0]  # Return first face encoding
        else:
            print("❌ No face encoding extracted")
            return None
            
    except Exception as e:
        print(f"❌ Error extracting face encoding: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_single_image(file_id, file_name, image_content):
    """Process a single image and extract face encodings"""
    try:
        # Convert to numpy array
        nparr = np.frombuffer(image_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return False
        
        # Convert BGR to RGB for DeepFace
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Try multiple face detection backends for better accuracy
        face_encodings_list = []
        
        # First try with opencv (faster)
        try:
            face_encodings_list = DeepFace.represent(
                img_path=img_rgb,
                model_name="Facenet",
                detector_backend="opencv"
            )
        except Exception as e:
            print(f"   ❌ OpenCV failed: {e}")
        
        # If no faces found, try with mtcnn (more accurate)
        if not face_encodings_list:
            try:
                face_encodings_list = DeepFace.represent(
                    img_path=img_rgb,
                    model_name="Facenet",
                    detector_backend="mtcnn"
                )
            except Exception as e:
                print(f"   ❌ MTCNN failed: {e}")
        
        # If still no faces, try with retinaface (most accurate)
        if not face_encodings_list:
            try:
                face_encodings_list = DeepFace.represent(
                    img_path=img_rgb,
                    model_name="Facenet",
                    detector_backend="retinaface"
                )
            except Exception as e:
                print(f"   ❌ RetinaFace failed: {e}")
        
        if face_encodings_list and len(face_encodings_list) > 0:
            # Store face encodings for this file
            face_encodings[file_id] = {
                'encodings': face_encodings_list,
                'count': len(face_encodings_list)
            }
            
            print(f"✅ {file_name}: {len(face_encodings_list)} faces detected and encoded")
            return True
        else:
            print(f"❌ {file_name}: No faces detected with any backend")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")
        return False

def load_images_from_drive():
    """Load images from Google Drive folder with incremental caching"""
    global face_encodings, file_metadata
    
    service = init_google_drive()
    if not service:
        return
    
    try:
        print("🔄 Starting incremental cache update...")
        
        # Get current files from Google Drive
        results = service.files().list(
            q=f"'{DRIVE_FOLDER_ID}' in parents and (mimeType contains 'image/')",
            fields="files(id,name,mimeType,modifiedTime)"
        ).execute()
        
        drive_files = results.get('files', [])
        print(f"📁 Found {len(drive_files)} images in Google Drive")
        
        if not drive_files:
            print("No images found in the specified folder")
            return
        
        # Track changes
        new_files = 0
        updated_files = 0
        deleted_files = 0
        unchanged_files = 0
        
        # Process each file
        for drive_file in drive_files:
            file_id = drive_file['id']
            file_name = drive_file['name']
            drive_modified_time = drive_file['modifiedTime']
            
            # Check if file exists in cache
            if file_id in file_metadata:
                cached_modified_time = file_metadata[file_id]['modified_time']
                
                # If timestamps match, skip processing
                if cached_modified_time == drive_modified_time:
                    unchanged_files += 1
                    continue
                else:
                    # File was updated, re-process it
                    print(f"🔄 Updating: {file_name} (modified)")
                    updated_files += 1
            else:
                # New file, process it
                print(f"🆕 New file: {file_name}")
                new_files += 1
            
            # Download and process the image
            try:
                request = service.files().get_media(fileId=file_id)
                image_content = request.execute()
                
                if process_single_image(file_id, file_name, image_content):
                    # Update metadata
                    file_metadata[file_id] = {
                        'name': file_name,
                        'id': file_id,
                        'faces_detected': face_encodings[file_id]['count'],
                        'modified_time': drive_modified_time
                    }
                else:
                    # No faces detected, still store metadata but mark as no faces
                    file_metadata[file_id] = {
                        'name': file_name,
                        'id': file_id,
                        'faces_detected': 0,
                        'modified_time': drive_modified_time
                    }
                    
            except Exception as e:
                print(f"❌ Error downloading {file_name}: {e}")
                continue
        
        # Check for deleted files
        cached_file_ids = set(file_metadata.keys())
        drive_file_ids = {f['id'] for f in drive_files}
        deleted_file_ids = cached_file_ids - drive_file_ids
        
        for deleted_id in deleted_file_ids:
            deleted_name = file_metadata[deleted_id]['name']
            print(f"��️ Deleted file: {deleted_name}")
            del file_metadata[deleted_id]
            if deleted_id in face_encodings:
                del face_encodings[deleted_id]
            deleted_files += 1
        
        # Save updated cache
        save_cache_to_disk()
        
        # Print summary
        print(f"\n📊 Cache Update Summary:")
        print(f"   ✅ Unchanged: {unchanged_files} files")
        print(f"   🆕 New: {new_files} files")
        print(f"   🔄 Updated: {updated_files} files")
        print(f"   ��️ Deleted: {deleted_files} files")
        print(f"   📸 Total with faces: {len([f for f in file_metadata.values() if f['faces_detected'] > 0])}")
        
    except Exception as e:
        print(f"❌ Error in incremental update: {e}")

def background_cache_updater():
    """Background thread to update cache every 5 minutes"""
    while True:
        try:
            time.sleep(300)  # Wait 5 minutes
            print("\n⏰ Running scheduled cache update...")
            load_images_from_drive()
        except Exception as e:
            print(f"❌ Error in background updater: {e}")

def find_matching_faces(selfie_encoding, threshold=0.4):
    """Find matching faces using DeepFace with improved accuracy"""
    matching_files = []
    
    if not selfie_encoding:
        return matching_files
    
    try:
        print(f"🔍 Selfie encoding type: {type(selfie_encoding)}")
        
        # Extract the actual embedding from DeepFace result
        if isinstance(selfie_encoding, dict) and 'embedding' in selfie_encoding:
            selfie_embedding = selfie_encoding['embedding']
            print(f"✅ Selfie embedding extracted: {len(selfie_embedding)} features")
        else:
            print(f"❌ Selfie encoding format unexpected: {selfie_encoding}")
            return matching_files
        
        # Store all similarities for debugging
        all_similarities = []
        
        for file_id, encoding_data in face_encodings.items():
            file_info = file_metadata.get(file_id, {})
            
            for stored_encoding in encoding_data['encodings']:
                try:
                    # Extract embedding from stored encoding
                    if isinstance(stored_encoding, dict) and 'embedding' in stored_encoding:
                        stored_embedding = stored_encoding['embedding']
                    else:
                        continue
                    
                    # Convert to numpy arrays for comparison
                    stored_array = np.array(stored_embedding, dtype=np.float32)
                    selfie_array = np.array(selfie_embedding, dtype=np.float32)
                    
                    # Use cosine similarity for better accuracy
                    stored_norm = np.linalg.norm(stored_array)
                    selfie_norm = np.linalg.norm(selfie_array)
                    
                    if stored_norm == 0 or selfie_norm == 0:
                        continue
                    
                    stored_normalized = stored_array / stored_norm
                    selfie_normalized = selfie_array / selfie_norm
                    
                    # Calculate cosine similarity
                    similarity = np.dot(stored_normalized, selfie_normalized)
                    
                    # Store similarity for debugging
                    all_similarities.append({
                        'file_name': file_info.get('name', 'Unknown'),
                        'similarity': similarity,
                        'file_id': file_id,
                        'faces_in_image': encoding_data['count']
                    })
                    
                    print(f"🔍 Comparing: {file_info.get('name', 'Unknown')} - Similarity: {similarity:.3f}")
                    
                    # Use multiple threshold levels for better accuracy
                    if similarity >= 0.7:  # High confidence match
                        confidence = "High"
                        match_type = "exact"
                    elif similarity >= 0.5:  # Medium confidence match
                        confidence = "Medium"
                        match_type = "likely"
                    elif similarity >= 0.4:  # Lower confidence but still possible
                        confidence = "Low"
                        match_type = "possible"
                    else:
                        continue
                    
                    matching_files.append({
                        'file_id': file_id,
                        'file_name': file_info.get('name', 'Unknown'),
                        'similarity_score': float(round(similarity, 3)),
                        'faces_in_image': encoding_data['count'],
                        'confidence': confidence,
                        'match_type': match_type
                    })
                    
                    print(f"✅ Match found: {file_info.get('name', 'Unknown')} with similarity {similarity:.3f} ({confidence} confidence)")
                    
                except Exception as e:
                    print(f"❌ Error comparing faces in {file_info.get('name', 'Unknown')}: {e}")
                    continue
        
        # Sort by similarity score (highest first)
        matching_files.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Print all similarities for debugging
        print("\n📊 ALL SIMILARITY SCORES:")
        all_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        for item in all_similarities:
            print(f"   {item['file_name']}: {item['similarity']:.3f}")
        
        print(f"\n✅ Found {len(matching_files)} matching images")
        print(f"🔍 Threshold used: {threshold}")
        
        return matching_files
        
    except Exception as e:
        print(f"❌ Error finding matching faces: {e}")
        import traceback
        traceback.print_exc()
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan_face():
    try:
        data = request.get_json()
        name = data.get('name', '')
        phone = data.get('phone', '')
        selfie_data = data.get('selfie', '')
        
        if not all([name, phone, selfie_data]):
            return jsonify({'error': 'Missing required data'}), 400
        
        # Save user data
        user_data = {
            'name': name,
            'phone': phone,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create user_data directory if it doesn't exist
        os.makedirs('user_data', exist_ok=True)
        
        # Save user data to file
        filename = f"user_data/user_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        with open(filename, 'w') as f:
            json.dump(user_data, f, indent=2)
        
        print(f"✅ User data saved to {filename}")
        print("🔍 Extracting face encoding from selfie...")
        
        # Extract face encoding from selfie
        selfie_encoding = extract_face_encoding(selfie_data)
        
        if not selfie_encoding:
            return jsonify({'error': 'No face detected in selfie. Please try again.'}), 400
        
        print("🔍 Finding matching faces...")
        
        # Find matching faces
        matching_files = find_matching_faces(selfie_encoding)
        
        if not matching_files:
            return jsonify({
                'message': 'No matching photos found',
                'user': name,
                'images_found': 0,
                'status': 'no_matches'
            })
        
        # Store results for results page
        session_results = {
            'user': name,
            'phone': phone,
            'matching_files': matching_files,
            'total_matches': len(matching_files)
        }
        
        # Save session results
        with open('session_results.json', 'w') as f:
            json.dump(session_results, f, indent=2)
        
        return jsonify({
            'message': f'Found {len(matching_files)} matching photos!',
            'user': name,
            'images_found': len(matching_files),
            'status': 'success',
            'redirect_url': '/results'
        })
        
    except Exception as e:
        print(f"❌ Error in scan_face: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/image/<file_id>')
def serve_image(file_id):
    """Serve image from Google Drive"""
    try:
        service = init_google_drive()
        if not service:
            return "Service not available", 500
        
        # Get file metadata
        file_metadata_info = file_metadata.get(file_id)
        if not file_metadata_info:
            return "File not found", 404
        
        # Download image content from Google Drive
        request = service.files().get_media(fileId=file_id)
        image_content = request.execute()
        
        # Determine content type based on file extension
        file_name = file_metadata_info['name'].lower()
        if file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
            content_type = 'image/jpeg'
        elif file_name.endswith('.png'):
            content_type = 'image/png'
        elif file_name.endswith('.webp'):
            content_type = 'image/webp'
        else:
            content_type = 'image/jpeg'  # Default
        
        # Create response with image content
        response = app.make_response(image_content)
        response.headers.set('Content-Type', content_type)
        response.headers.set('Cache-Control', 'public, max-age=3600')  # Cache for 1 hour
        
        return response
        
    except Exception as e:
        print(f"❌ Error serving image {file_id}: {e}")
        return "Error loading image", 500

@app.route('/download/<file_id>')
def download_image(file_id):
    """Download image from Google Drive"""
    try:
        service = init_google_drive()
        if not service:
            return "Service not available", 500
        
        # Get file metadata
        file_metadata_info = file_metadata.get(file_id)
        if not file_metadata_info:
            return "File not found", 404
        
        # Download image content from Google Drive
        request = service.files().get_media(fileId=file_id)
        image_content = request.execute()
        
        # Create response for download
        response = app.make_response(image_content)
        response.headers.set('Content-Type', 'application/octet-stream')
        response.headers.set('Content-Disposition', 'attachment', filename=file_metadata_info['name'])
        
        return response
        
    except Exception as e:
        print(f"❌ Error downloading image {file_id}: {e}")
        return "Error downloading image", 500

@app.route('/results')
def results():
    try:
        # Load session results
        if os.path.exists('session_results.json'):
            with open('session_results.json', 'r') as f:
                session_data = json.load(f)
            return render_template('results.html', 
                                user=session_data.get('user', 'Guest'),
                                matching_files=session_data.get('matching_files', []),
                                total_matches=session_data.get('total_matches', 0))
        else:
            return render_template('results.html', 
                                user='Guest',
                                matching_files=[],
                                total_matches=0)
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return render_template('results.html', 
                            user='Guest',
                            matching_files=[],
                            total_matches=0)

@app.route('/cache-status')
def cache_status():
    """Show cache status for debugging"""
    return jsonify({
        'total_files': len(file_metadata),
        'files_with_faces': len([f for f in file_metadata.values() if f['faces_detected'] > 0]),
        'cache_last_updated': cache_last_updated,
        'face_encodings_count': len(face_encodings)
    })

if __name__ == '__main__':
    print("�� Starting Optimized DeepFace Photo Finder App...")
    
    # Load existing cache first
    load_cache_from_disk()
    
    # Initial load/update of images
    load_images_from_drive()
    
    print(f"📸 Loaded {len(file_metadata)} images with faces")
    print("🌐 Starting web server...")
    
    # Start background cache updater thread
    cache_thread = threading.Thread(target=background_cache_updater, daemon=True)
    cache_thread.start()
    print("🔄 Background cache updater started (runs every 5 minutes)")
    
    app.run(debug=True, host='0.0.0.0', port=5000)