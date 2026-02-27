# 👁️ VisionVoice — AI Image Describer for the Visually Impaired

A full-stack college AI project that lets users upload an image and **hear a spoken description** of what's in it — powered by BLIP (Salesforce) for image captioning and gTTS for text-to-speech.

---

## 🗂️ Project Structure

```
visionvoice/
├── backend/
│   ├── app.py              ← Flask API server (main entry point)
│   ├── model_loader.py     ← Loads and runs the BLIP AI model
│   ├── tts_generator.py    ← Converts text to MP3 using gTTS
│   ├── requirements.txt    ← Python dependencies
│   └── static/
│       └── audio/          ← Generated MP3 files saved here
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx           ← Root component + API call logic
        ├── App.css           ← All styles
        └── components/
            ├── ImageUpload.jsx      ← Drag & drop uploader + preview
            ├── DescriptionResult.jsx ← Shows description + audio player
            └── AudioPlayer.jsx      ← Custom audio controls
```

---

## ⚙️ How to Set Up & Run Locally

### Prerequisites
- **Python 3.9+** installed
- **Node.js 18+** installed
- Internet connection (model downloads ~1GB on first run)

---

### Step 1 — Set up the Python Backend

Open a terminal and navigate to the backend folder:

```bash
cd visionvoice/backend
```

Create a virtual environment (recommended):

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Flask server:

```bash
python app.py
```

You should see:
```
Loading BLIP model... (this may take a minute on first run)
Model loaded successfully on cpu!
Backend ready! Listening for requests...
 * Running on http://0.0.0.0:5000
```

> ⏳ **First run takes 1–2 minutes** to download the BLIP model (~1GB). After that it loads in seconds from cache.

---

### Step 2 — Set up the React Frontend

Open a **second terminal** and navigate to the frontend folder:

```bash
cd visionvoice/frontend
```

Install Node.js dependencies:

```bash
npm install
```

Start the React development server:

```bash
npm run dev
```

You should see:
```
  VITE v5.x.x  ready

  ➜  Local:   http://localhost:3000/
```

---

### Step 3 — Use the App

1. Open your browser and go to **http://localhost:3000**
2. Upload any image by dragging & dropping or clicking the upload area
3. Click **"Describe Image"**
4. Wait 5–15 seconds for the AI to analyze it
5. Read the description and click ▶ to **hear it spoken aloud**

---

## 🤖 How the AI Works

```
User uploads image
        ↓
Frontend (React) sends image via POST /describe-image
        ↓
Flask backend receives image
        ↓
BLIP model (Salesforce/blip-image-captioning-base) generates text description
        ↓
gTTS converts text → MP3 audio file
        ↓
Flask returns { description, audio_url }
        ↓
Frontend displays text + plays audio automatically
```

---

## 🧰 Tech Stack

| Layer       | Technology                                      |
|-------------|------------------------------------------------|
| Frontend    | React 18, Vite, CSS3                           |
| Backend     | Python, Flask, Flask-CORS                      |
| AI Model    | BLIP (Salesforce/blip-image-captioning-base)   |
| AI Library  | HuggingFace Transformers, PyTorch              |
| Image Proc. | Pillow (PIL)                                   |
| Speech      | gTTS (Google Text-to-Speech)                   |
| API Style   | REST (JSON over HTTP)                          |

---

## 🎓 Demo Tips (For Class Presentation)

1. **Pre-load the model** — Start the backend 5 minutes before your demo so the model is warmed up
2. **Use clear images** — A dog in a park, a person cooking, a car on a road all work great
3. **Show the flow** — Open DevTools Network tab to show the actual POST request happening
4. **Accessibility angle** — Point out the high-contrast design, large buttons, and keyboard navigation
5. **Offline audio** — The "Download Audio" button lets you save and play the MP3 offline

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|---------|
| Model download stuck | Check your internet connection; it's ~1GB |
| `CORS error` in browser | Make sure Flask is running on port 5000 |
| Audio won't play | Browser may block autoplay; click the ▶ button manually |
| `ModuleNotFoundError` | Make sure you activated the virtual environment |
| Slow first response | Normal — BLIP model needs ~5–15 sec on CPU |

---

## 📚 Acknowledgements

- [BLIP Model](https://huggingface.co/Salesforce/blip-image-captioning-base) by Salesforce Research
- [gTTS](https://gtts.readthedocs.io/) by Pierre Nicolas Durette
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
