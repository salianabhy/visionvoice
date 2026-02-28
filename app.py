# app.py
# VisionVoice Flask backend.
# POST /describe-image  — receives image, returns description + hazard + audio URL
# GET  /               — health check

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import os, io, traceback

from model_loader import load_model, generate_caption, check_for_hazards
from tts_generator import generate_audio, cleanup_old_audio

app = Flask(__name__)

CORS(
    app,
    resources={r"/*": {"origins": [
        "http://localhost:3000",                        # local dev
        "https://visionvoicee.vercel.app",      # production
        "https://*.vercel.app",                         # all Vercel preview URLs
    ]}},
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"],
)

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

_model_loaded = False
_model_error  = None


def ensure_model_loaded():
    global _model_loaded, _model_error
    if _model_loaded:
        return
    if _model_error:
        raise RuntimeError(f"Model init failed earlier: {_model_error}")
    try:
        load_model()
        _model_loaded = True
        print("Model ready.")
    except Exception as e:
        _model_error = str(e)
        raise


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status":       "VisionVoice API is running ✓",
        "model_loaded": _model_loaded,
        "model_error":  _model_error,
    })


@app.route("/describe-image", methods=["POST", "OPTIONS"])
def describe_image():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    # 1. Init
    try:
        ensure_model_loaded()
    except Exception as e:
        return jsonify({"error": f"Init failed: {str(e)}"}), 500

    # 2. Validate file
    if "image" not in request.files:
        return jsonify({"error": "No image field in request."}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        # 3. Open image
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        print(f"Image: {image.size[0]}x{image.size[1]}px")

        # 4. Generate caption via HuggingFace API
        print("Generating caption...")
        raw_caption = generate_caption(image)
        print(f"Raw caption: {raw_caption}")

        # 5. Format description
        if raw_caption:
            c = raw_caption[0].lower()
            description = f"This image shows {c}{raw_caption[1:]}"
            if not description.endswith("."):
                description += "."
        else:
            description = "The image could not be described."

        print(f"Description: {description}")

        # 6. Hazard scan
        hazard = check_for_hazards(image, scene_description=description)

        # 7. Generate audio
        audio_filename = generate_audio(description, AUDIO_DIR)
        cleanup_old_audio(AUDIO_DIR, keep_latest=10)

        # 8. Return
        return jsonify({
            "description": description,
            "audio_url":   f"/static/audio/{audio_filename}",
            "hazard":      hazard,
        })

    except Exception as e:
        print(f"ERROR:\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/static/audio/<filename>", methods=["GET"])
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename, mimetype="audio/mpeg")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting VisionVoice on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
