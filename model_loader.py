# model_loader.py
# Uses huggingface_hub InferenceClient — the official library that handles
# all API routing, authentication and retries automatically.
# No manual URL needed — the library knows the correct endpoint.

import os
import io
from PIL import Image
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_API_TOKEN", "")
_client  = None   # created once on first request


def _get_client():
    global _client
    if _client is None:
        if not HF_TOKEN:
            raise RuntimeError(
                "HF_API_TOKEN not set. "
                "Add it in Render → Environment → HF_API_TOKEN."
            )
        _client = InferenceClient(token=HF_TOKEN)
        print("HuggingFace InferenceClient ready.")
    return _client


def load_model():
    """Validate token on first request. No local model to load."""
    _get_client()
    print("HuggingFace API mode — model runs on HF servers.")


def generate_caption(image: Image.Image) -> str:
    """
    Send image to HuggingFace Inference API using the official client.
    Uses BLIP-large running on HF's GPU servers.
    """
    client = _get_client()

    # Convert PIL image to JPEG bytes
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    buf.seek(0)

    print(f"Sending image to HuggingFace ({buf.getbuffer().nbytes // 1024}KB)...")

    try:
        result = client.image_to_text(
            buf,
            model="Salesforce/blip-image-captioning-large",
        )
        print(f"HF raw result: {result}")
    except Exception as e:
        raise RuntimeError(f"HuggingFace API call failed: {str(e)}")

    # result is a string or object with .generated_text
    if hasattr(result, "generated_text"):
        caption = result.generated_text
    elif isinstance(result, str):
        caption = result
    else:
        caption = str(result)

    caption = caption.strip()
    if caption and not caption[0].isupper():
        caption = caption[0].upper() + caption[1:]
    if caption and not caption.endswith("."):
        caption += "."

    print(f"Caption: {caption}")
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
