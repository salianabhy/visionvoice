# model_loader.py
# Instead of loading BLIP locally (requires 1.5-4GB RAM, crashes Render free tier),
# we call the HuggingFace Inference API — the model runs on HF's servers,
# Render just forwards the image and returns the result.
# RAM usage on Render: ~50MB instead of ~1500MB.

import os
import io
import requests
from PIL import Image

# HuggingFace Inference API endpoint for BLIP
HF_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"

# Read token from environment variable (set in Render dashboard)
HF_TOKEN = os.environ.get("HF_API_TOKEN", "")


def _get_headers():
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_API_TOKEN environment variable is not set. "
            "Add it in Render → Environment."
        )
    return {"Authorization": f"Bearer {HF_TOKEN}"}


def load_model():
    """
    No-op — model runs on HuggingFace servers, nothing to load locally.
    Kept for API compatibility with app.py which calls this on first request.
    """
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_API_TOKEN not set. Add your HuggingFace token in "
            "Render → Environment → HF_API_TOKEN."
        )
    print("HuggingFace Inference API mode — no local model to load.")
    print(f"Using model: {HF_API_URL}")


def generate_caption(image: Image.Image) -> str:
    """
    Send image to HuggingFace Inference API and get back a caption.

    HuggingFace accepts raw image bytes (JPEG).
    Returns a clean, capitalised sentence.
    """
    # Convert PIL image to JPEG bytes
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    image_bytes = buf.getvalue()

    print(f"Sending {len(image_bytes)//1024}KB image to HuggingFace API...")

    try:
        response = requests.post(
            HF_API_URL,
            headers=_get_headers(),
            data=image_bytes,
            timeout=60,  # HF cold start can take ~20-30s on first request
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("HuggingFace API timed out. Try again in a moment.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Could not connect to HuggingFace API. Check internet connection.")

    # Handle HuggingFace-specific errors
    if response.status_code == 503:
        # Model is loading on HF side — retry after a few seconds
        raise RuntimeError(
            "HuggingFace model is loading (503). "
            "This takes ~20 seconds on first use. Please try again."
        )
    if response.status_code == 401:
        raise RuntimeError("Invalid HF_API_TOKEN. Check your token in Render environment variables.")

    if not response.ok:
        raise RuntimeError(
            f"HuggingFace API error {response.status_code}: {response.text[:200]}"
        )

    result = response.json()
    print(f"HuggingFace raw response: {result}")

    # HF returns: [{"generated_text": "a person sitting at a desk"}]
    if isinstance(result, list) and result:
        caption = result[0].get("generated_text", "")
    elif isinstance(result, dict):
        caption = result.get("generated_text", "")
    else:
        caption = str(result)

    # Clean up and capitalise
    caption = caption.strip()
    if caption and not caption[0].isupper():
        caption = caption[0].upper() + caption[1:]
    if caption and not caption.endswith("."):
        caption += "."

    print(f"Final caption: {caption}")
    return caption


# ── Hazard detection — unchanged, scans caption text ─────────────────────────
# (No API call needed — we just scan the caption BLIP already returned)

HAZARD_KEYWORDS = {
    # PRIORITY 1: CRITICAL
    "fire":           (1, "fire detected — move away immediately",    "🔥"),
    "flame":          (1, "fire detected — move away immediately",    "🔥"),
    "flames":         (1, "fire detected — move away immediately",    "🔥"),
    "burning":        (1, "fire detected — move away immediately",    "🔥"),
    "smoke":          (1, "smoke detected — possible fire nearby",    "💨"),
    "explosion":      (1, "explosion risk — move away",               "💥"),
    "electric":       (1, "electrical hazard nearby",                 "⚡"),
    "electrical":     (1, "electrical hazard nearby",                 "⚡"),
    "sparks":         (1, "electrical sparks — do not touch",         "⚡"),
    "chemical":       (1, "chemical hazard nearby",                   "☣️"),
    "toxic":          (1, "toxic material nearby",                    "☣️"),
    "gun":            (1, "weapon detected nearby",                   "🚨"),
    "weapon":         (1, "weapon detected nearby",                   "🚨"),
    "knife":          (1, "sharp weapon nearby",                      "🚨"),
    "flood":          (1, "flooding detected — avoid area",           "🌊"),

    # PRIORITY 2: SERIOUS
    "car":            (2, "vehicle nearby — stop and wait",           "🚗"),
    "truck":          (2, "large vehicle nearby — stop and wait",     "🚛"),
    "bus":            (2, "bus nearby — stop and wait",               "🚌"),
    "van":            (2, "vehicle nearby — stop and wait",           "🚗"),
    "motorcycle":     (2, "motorcycle nearby — be careful",           "🏍️"),
    "motorbike":      (2, "motorcycle nearby — be careful",           "🏍️"),
    "vehicle":        (2, "vehicle nearby — stop and wait",           "🚗"),
    "traffic":        (2, "traffic ahead — do not cross",             "🚦"),
    "road":           (2, "road ahead — watch for vehicles",          "🛣️"),
    "street":         (2, "street ahead — watch for traffic",         "🛣️"),
    "train":          (2, "train nearby — stay clear of tracks",      "🚆"),
    "track":          (2, "train track ahead — cross carefully",      "🚆"),
    "crowd":          (2, "crowd ahead — move carefully",             "👥"),

    # PRIORITY 3: HIGH
    "stair":          (3, "stairs ahead — hold the railing",          "🪜"),
    "stairs":         (3, "stairs ahead — hold the railing",          "🪜"),
    "staircase":      (3, "staircase ahead — hold the railing",       "🪜"),
    "stairway":       (3, "stairway ahead — hold the railing",        "🪜"),
    "step":           (3, "step ahead — watch your footing",          "⚠️"),
    "steps":          (3, "steps ahead — watch your footing",         "⚠️"),
    "escalator":      (3, "escalator ahead — hold the railing",       "🪜"),
    "ladder":         (3, "ladder nearby — be careful",               "🪜"),
    "ramp":           (3, "ramp ahead — uneven surface",              "⚠️"),
    "cliff":          (3, "drop ahead — stay back",                   "🏔️"),
    "ledge":          (3, "ledge ahead — stay back",                  "⚠️"),
    "hole":           (3, "hole in floor — do not step",              "⚠️"),
    "gap":            (3, "gap ahead — do not step",                  "⚠️"),
    "pit":            (3, "pit ahead — do not step forward",          "⚠️"),
    "wet":            (3, "wet surface — slip risk",                  "💧"),
    "slippery":       (3, "slippery surface — slow down",             "💧"),
    "puddle":         (3, "puddle on ground",                         "💧"),
    "ice":            (3, "ice on ground — slip risk",                "🧊"),
    "icy":            (3, "icy surface — slip risk",                  "🧊"),
    "snow":           (3, "snow on ground — slippery",                "❄️"),

    # PRIORITY 4: MEDIUM
    "door":           (4, "door ahead",                               "🚪"),
    "doorway":        (4, "doorway ahead",                            "🚪"),
    "entrance":       (4, "entrance ahead",                           "🚪"),
    "exit":           (4, "exit ahead",                               "🚪"),
    "gate":           (4, "gate ahead",                               "🚧"),
    "wall":           (4, "wall ahead — stop",                        "🧱"),
    "fence":          (4, "fence ahead",                              "🚧"),
    "barrier":        (4, "barrier ahead",                            "🚧"),
    "pole":           (4, "pole in path",                             "⚠️"),
    "pillar":         (4, "pillar ahead",                             "⚠️"),
    "column":         (4, "column ahead",                             "⚠️"),
    "construction":   (4, "construction zone — be careful",           "🏗️"),
    "dog":            (4, "dog nearby — approach carefully",          "🐕"),
    "animal":         (4, "animal nearby — be cautious",              "🐾"),
    "snake":          (4, "snake nearby — do not approach",           "🐍"),
    "person":         (4, "person directly ahead — slow down",        "🧍"),
    "people":         (4, "people ahead — slow down",                 "👥"),
    "child":          (4, "child nearby — be extra careful",          "👶"),
    "glass":          (4, "glass nearby — be careful",                "⚠️"),
    "broken":         (4, "broken object nearby",                     "⚠️"),
    "sharp":          (4, "sharp object nearby",                      "⚠️"),
    "debris":         (4, "debris on ground",                         "⚠️"),

    # PRIORITY 5: LOW
    "chair":          (5, "chair in path",                            "🪑"),
    "table":          (5, "table ahead",                              "🪑"),
    "desk":           (5, "desk ahead",                               "🪑"),
    "box":            (5, "box in path",                              "📦"),
    "cord":           (5, "cord on ground — trip hazard",             "⚠️"),
    "cable":          (5, "cable on ground — trip hazard",            "⚠️"),
    "wire":           (5, "wire on ground — trip hazard",             "⚠️"),
    "rug":            (5, "rug on floor — edge risk",                 "⚠️"),
    "carpet":         (5, "carpet edge — trip risk",                  "⚠️"),
    "clutter":        (5, "clutter on floor",                         "⚠️"),
}


def check_for_hazards(image: Image.Image, scene_description: str = "") -> dict:
    """Scan caption text for highest-priority hazard keyword."""
    if not scene_description:
        return {
            "hazard_detected": False,
            "hazard_type":     "",
            "hazard_emoji":    "",
            "hazard_priority": 99,
            "matched_keyword": "",
        }

    text = scene_description.lower()
    print(f"[Hazard scan] '{text}'")

    best = (99, "", "", "")  # priority, label, emoji, keyword

    for keyword, (priority, label, emoji) in HAZARD_KEYWORDS.items():
        if keyword in text and priority < best[0]:
            best = (priority, label, emoji, keyword)

    if best[1]:
        print(f"[Hazard] priority={best[0]} '{best[3]}' → '{best[1]}'")
    else:
        print("[Hazard] none found")

    return {
        "hazard_detected": bool(best[1]),
        "hazard_type":     best[1],
        "hazard_emoji":    best[2],
        "hazard_priority": best[0],
        "matched_keyword": best[3],
    }
