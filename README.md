# Wedding Face Recognition App

A web application that helps users find their photos from a Google Drive folder by scanning their face from a selfie. Built with Flask, DeepFace, and Google Drive API.

---

## Key Features

- **Face Search:** Upload a selfie and instantly find matching photos from a Google Drive folder.
- **Multi-Face Detection:** Detects and encodes multiple faces in group photos.
- **Accurate Matching:** Uses DeepFace and cosine similarity for robust face recognition.
- **Automatic Updates:** Background process keeps photo and face data up-to-date.
- **Fast Results:** Optimized caching and asynchronous processing for quick searches.
- **User-Friendly Interface:** Modern, responsive web design for easy use.
- **Download Photos:** Download matched images directly from the results page.

---

## Project Structure

```
wed/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── credentials.json        # Google Drive API credentials (not tracked in git)
├── cache/                  # Stores face encodings and metadata
│   ├── face_encodings_cache.pkl
│   ├── file_metadata_cache.json
│   └── cache_timestamp.txt
├── templates/
│   ├── index.html          # Upload/selfie page
│   └── results.html        # Results display page
└── session_results.json    # Temporary session storage
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd wed
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Google Drive Credentials

- Place your `credentials.json` (Google service account) in the project root.
- Share your Google Drive folder with the service account email.

### 4. Run the Application

```bash
python app.py
```

- Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## Usage

1. **Open the app in your browser.**
2. **Enter your name and phone number.**
3. **Take or upload a selfie.**
4. **Submit to search for matching photos.**
5. **View and download your matched photos from the results page.**

---

## Deployment (Render)

1. Push your code to GitHub.
2. Create a new Web Service on [Render](https://render.com/).
3. Set the start command to `python app.py`.
4. Add your `requirements.txt` and `credentials.json` (as a secret or environment variable).
5. Deploy and access your app via Render’s URL.

---

## Optimization & Performance

- **Caching:** Face encodings and metadata are cached for fast lookups.
- **Asynchronous Processing:** Uses background threads for cache updates and can be extended with async image processing.
- **Efficient Matching:** Only compares faces using optimized numpy and scipy operations.
- **Multi-Face Support:** Each image can store multiple face encodings for group photo matching.

---

## Troubleshooting

- **New photos not detected:** Restart the app or use the `/force-cache-update` route.
- **No matches found:** Ensure your selfie is clear and matches faces in Drive photos.
- **Google Drive issues:** Check folder permissions and service account access.

---

## Contributing

Pull requests and suggestions are welcome! Please open an issue for bugs or feature requests.

---

## License

MIT License

---

## Credits

- [Flask](https://flask.palletsprojects.com/)
- [DeepFace](https://github.com/serengil/deepface)
- [Google Drive API](https://developers.google.com/drive)
