# app.py
# Main Flask backend for VisionVoice.
# Exposes a single REST API endpoint: POST /describe-image
# The frontend calls this with an uploaded image and receives
# a text description + audio file URL in return.

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import os
import io
import traceback

from model_loader import load_model, generate_caption, check_for_hazards
from tts_generator import generate_audio, cleanup_old_audio

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── CRITICAL: CORS must be fully explicit to handle browser preflight requests
# Browsers send an OPTIONS "preflight" request before POST — if CORS doesn't
# respond to it correctly, the browser blocks the actual request entirely,
# which shows up as "Failed to fetch" with no useful error message.
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
    supports_credentials=False,
)

# Directory where generated audio files will be saved and served
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# ── Lazy Model Loading ────────────────────────────────────────────────────────
# We do NOT load the model at startup — if it crashes, Flask never starts
# and the frontend sees "Failed to fetch" with no explanation.
# Instead we load on the first real request so Flask always starts cleanly.
_model_loaded = False
_model_error = None

def ensure_model_loaded():
    """Load the BLIP model on first use. Caches result for all future requests."""
    global _model_loaded, _model_error
    if _model_loaded:
        return  # Already loaded, nothing to do
    if _model_error:
        raise RuntimeError(f"Model failed to load earlier: {_model_error}")
    try:
        print("Loading BLIP model for first request (may take 1-2 min)...")
        load_model()
        _model_loaded = True
        print("Model loaded OK!")
    except Exception as e:
        _model_error = str(e)
        raise


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health_check():
    """
    Health check endpoint — open http://localhost:5001 in a browser
    to confirm the backend is running before starting the frontend.
    """
    return jsonify({
        "status": "VisionVoice API is running ✓",
        "model_loaded": _model_loaded,
        "model_error": _model_error,
    })


@app.route("/describe-image", methods=["POST", "OPTIONS"])
def describe_image():
    """
    Main endpoint: accepts an image, returns description + audio URL.

    Request:  multipart/form-data with field 'image' containing the image file
    Response: JSON { "description": "...", "audio_url": "/static/audio/file.mp3" }

    The OPTIONS handler is needed for CORS preflight — browsers send this
    automatically before the actual POST request.
    """
    # Handle CORS preflight (OPTIONS) — flask-cors should do this automatically
    # but we add it explicitly as a safety net
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight ok"}), 200

    # ── 1. Load model on first request ───────────────────────────────────────
    try:
        ensure_model_loaded()
    except Exception as e:
        print(f"Model load error: {e}")
        return jsonify({
            "error": (
                f"AI model failed to load: {str(e)}. "
                "Make sure PyTorch and Transformers are installed correctly."
            )
        }), 500

    # ── 2. Validate that an image was sent ───────────────────────────────────
    if "image" not in request.files:
        return jsonify({
            "error": "No image file found in request. The file field must be named 'image'."
        }), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "Empty filename. Please select a valid image."}), 400

    # Basic file type validation
    allowed_extensions = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
    file_ext = image_file.filename.rsplit(".", 1)[-1].lower() if "." in image_file.filename else ""
    if file_ext not in allowed_extensions:
        return jsonify({
            "error": f"Unsupported file type '{file_ext}'. Please upload a PNG, JPG, or WEBP image."
        }), 400

    try:
        # ── 3. Open the image with Pillow ─────────────────────────────────────
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print(f"Image received: {image.size[0]}×{image.size[1]} px")

        # ── 4. Generate AI caption via BLIP ──────────────────────────────────
        print("Generating caption...")
        raw_caption = generate_caption(image)
        print(f"Raw caption: {raw_caption}")

        # Format into a natural sentence for visually impaired users
        if raw_caption:
            first_char = raw_caption[0].lower()
            rest = raw_caption[1:]
            description = f"This image shows {first_char}{rest}"
            if not description.endswith("."):
                description += "."
        else:
            description = "The image could not be described. Please try a different photo."

        print(f"Final description: {description}")

        # ── 5. Hazard detection — scan the description for danger words ─────────
        # No extra model call needed. The caption BLIP just generated already
        # contains the object names accurately. We just scan that text.
        print("Running hazard scan...")
        hazard_result = check_for_hazards(image, scene_description=description)
        print(f"Hazard result: {hazard_result}")

        # ── 6. Convert description to speech using gTTS ───────────────────────
        print("Generating audio...")
        audio_filename = generate_audio(description, AUDIO_DIR)
        cleanup_old_audio(AUDIO_DIR, keep_latest=10)

        # ── 7. Return everything to frontend ──────────────────────────────────
        return jsonify({
            "description": description,
            "audio_url":   f"/static/audio/{audio_filename}",
            "hazard":      hazard_result,  # { hazard_detected, hazard_type, raw_answer }
        })

    except Exception as e:
        print(f"ERROR processing image:\n{traceback.format_exc()}")
        return jsonify({
            "error": f"Processing failed: {str(e)}"
        }), 500


@app.route("/static/audio/<filename>", methods=["GET"])
def serve_audio(filename):
    """Serve generated MP3 audio files to the frontend."""
    return send_from_directory(AUDIO_DIR, filename, mimetype="audio/mpeg")


# ── Run Server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  VisionVoice Backend starting...")
    print("  Open http://localhost:5001 to confirm it's running")
    print("  Then open http://localhost:3000 for the frontend")
    print("=" * 55)
    # debug=False for stability; host=0.0.0.0 so any device on your network can reach it
    app.run(host="0.0.0.0", port=5001, debug=False)