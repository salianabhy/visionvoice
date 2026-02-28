# model_loader.py
# Calls HuggingFace Inference API using raw HTTP requests (no SDK needed).
# The model runs on HuggingFace's servers — Render only needs ~50MB RAM.
#
# SETUP: Add HF_API_TOKEN in Render → Environment before deploying.
# Get token free at: huggingface.co → Settings → Access Tokens → New Token (Read)

import os
import io
import time
import requests
from PIL import Image

HF_TOKEN   = os.environ.get("HF_API_TOKEN", "")
HF_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"


def load_model():
    """Validate token exists. No local model to load — runs on HF servers."""
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_API_TOKEN not set. "
            "Go to Render → your service → Environment → Add HF_API_TOKEN."
        )
    print("HuggingFace API mode ready. Token found.")


def generate_caption(image: Image.Image) -> str:
    """
    Send image to HuggingFace Inference API, receive caption text.
    Retries up to 3 times to handle 503 model-loading delays on HF side.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set in Render environment.")

    # Resize to max 512px — speeds up transfer, HF resizes anyway
    MAX_DIM = 512
    w, h = image.size
    if w > MAX_DIM or h > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Convert PIL image → JPEG bytes
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    image_bytes = buf.getvalue()
    print(f"Sending {len(image_bytes) // 1024}KB to HuggingFace...")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type":  "image/jpeg",
    }

    for attempt in range(1, 4):   # try up to 3 times
        try:
            resp = requests.post(
                HF_API_URL,
                headers=headers,
                data=image_bytes,
                timeout=60,
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("HuggingFace API timed out after 60s. Try again.")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Cannot reach HuggingFace API: {e}")

        print(f"HF response (attempt {attempt}): {resp.status_code} — {resp.text[:120]}")

        if resp.status_code == 200:
            break   # success

        if resp.status_code == 503:
            # Model is loading on HF side — normal on first use
            wait = 20 * attempt
            print(f"HF model loading, waiting {wait}s before retry...")
            time.sleep(wait)
            continue

        if resp.status_code == 401:
            raise RuntimeError(
                "HuggingFace token rejected (401). "
                "Check HF_API_TOKEN value in Render → Environment."
            )

        # Any other error — raise immediately
        raise RuntimeError(
            f"HuggingFace API returned {resp.status_code}: {resp.text[:300]}"
        )
    else:
        raise RuntimeError(
            "HuggingFace model did not respond after 3 attempts. "
            "Wait a minute and try again."
        )

    # Parse response — HF returns: [{"generated_text": "a person sitting..."}]
    result = resp.json()
    print(f"HF result: {result}")

    if isinstance(result, list) and result:
        caption = result[0].get("generated_text", "")
    elif isinstance(result, dict):
        caption = result.get("generated_text", "")
    else:
        caption = str(result)

    # Clean up text
    caption = caption.strip()
    if caption and not caption[0].isupper():
        caption = caption[0].upper() + caption[1:]
    if caption and not caption.endswith("."):
        caption += "."

    print(f"Caption: {caption}")
    return caption


# ── Hazard detection — scans caption text for danger keywords ─────────────────
# Priority levels: 1=critical, 2=serious, 3=high, 4=medium, 5=low
# Returns the HIGHEST priority (lowest number) hazard found.

HAZARD_KEYWORDS = {
    # PRIORITY 1 — CRITICAL
    "fire":        (1, "fire detected — move away immediately", "🔥"),
    "flame":       (1, "fire detected — move away immediately", "🔥"),
    "flames":      (1, "fire detected — move away immediately", "🔥"),
    "burning":     (1, "fire detected — move away immediately", "🔥"),
    "smoke":       (1, "smoke detected — possible fire nearby", "💨"),
    "explosion":   (1, "explosion risk — move away",            "💥"),
    "electric":    (1, "electrical hazard nearby",              "⚡"),
    "electrical":  (1, "electrical hazard nearby",              "⚡"),
    "sparks":      (1, "electrical sparks — do not touch",      "⚡"),
    "chemical":    (1, "chemical hazard nearby",                "☣️"),
    "toxic":       (1, "toxic material nearby",                 "☣️"),
    "gun":         (1, "weapon detected nearby",                "🚨"),
    "weapon":      (1, "weapon detected nearby",                "🚨"),
    "knife":       (1, "sharp weapon nearby",                   "🚨"),
    "flood":       (1, "flooding detected — avoid area",        "🌊"),
    "flooded":     (1, "flooding detected — avoid area",        "🌊"),

    # PRIORITY 2 — SERIOUS
    "car":         (2, "vehicle nearby — stop and wait",        "🚗"),
    "truck":       (2, "large vehicle nearby",                  "🚛"),
    "bus":         (2, "bus nearby — stop and wait",            "🚌"),
    "van":         (2, "vehicle nearby — stop and wait",        "🚗"),
    "motorcycle":  (2, "motorcycle nearby — be careful",        "🏍️"),
    "motorbike":   (2, "motorcycle nearby — be careful",        "🏍️"),
    "vehicle":     (2, "vehicle nearby — stop and wait",        "🚗"),
    "traffic":     (2, "traffic ahead — do not cross",          "🚦"),
    "road":        (2, "road ahead — watch for vehicles",       "🛣️"),
    "street":      (2, "street ahead — watch for traffic",      "🛣️"),
    "train":       (2, "train nearby — stay clear of tracks",   "🚆"),
    "track":       (2, "train track — cross carefully",         "🚆"),
    "crowd":       (2, "crowd ahead — move carefully",          "👥"),

    # PRIORITY 3 — HIGH
    "stair":       (3, "stairs ahead — hold the railing",       "🪜"),
    "stairs":      (3, "stairs ahead — hold the railing",       "🪜"),
    "staircase":   (3, "staircase ahead — hold the railing",    "🪜"),
    "stairway":    (3, "stairway ahead — hold the railing",     "🪜"),
    "step":        (3, "step ahead — watch your footing",       "⚠️"),
    "steps":       (3, "steps ahead — watch your footing",      "⚠️"),
    "escalator":   (3, "escalator ahead — hold the railing",    "🪜"),
    "ladder":      (3, "ladder nearby — be careful",            "🪜"),
    "ramp":        (3, "ramp ahead — uneven surface",           "⚠️"),
    "cliff":       (3, "drop ahead — stay back",                "🏔️"),
    "ledge":       (3, "ledge ahead — stay back",               "⚠️"),
    "drop":        (3, "drop ahead — stay back",                "⚠️"),
    "pit":         (3, "pit ahead — do not step forward",       "⚠️"),
    "hole":        (3, "hole in floor — do not step",           "⚠️"),
    "gap":         (3, "gap ahead — do not step",               "⚠️"),
    "ditch":       (3, "ditch ahead — step carefully",          "⚠️"),
    "manhole":     (3, "manhole ahead — avoid",                 "⚠️"),
    "wet":         (3, "wet surface — slip risk",               "💧"),
    "slippery":    (3, "slippery surface — slow down",          "💧"),
    "puddle":      (3, "puddle on ground",                      "💧"),
    "spill":       (3, "spill on floor — slip risk",            "💧"),
    "ice":         (3, "ice on ground — slip risk",             "🧊"),
    "icy":         (3, "icy surface — slip risk",               "🧊"),
    "snow":        (3, "snow on ground — slippery",             "❄️"),
    "mud":         (3, "muddy ground — slippery",               "⚠️"),

    # PRIORITY 4 — MEDIUM
    "door":        (4, "door ahead",                            "🚪"),
    "doorway":     (4, "doorway ahead",                         "🚪"),
    "entrance":    (4, "entrance ahead",                        "🚪"),
    "exit":        (4, "exit ahead",                            "🚪"),
    "gate":        (4, "gate ahead",                            "🚧"),
    "turnstile":   (4, "turnstile ahead",                       "🚧"),
    "wall":        (4, "wall ahead — stop",                     "🧱"),
    "fence":       (4, "fence ahead",                           "🚧"),
    "barrier":     (4, "barrier ahead",                         "🚧"),
    "bollard":     (4, "bollard in path",                       "🚧"),
    "pole":        (4, "pole in path",                          "⚠️"),
    "pillar":      (4, "pillar ahead",                          "⚠️"),
    "column":      (4, "column ahead",                          "⚠️"),
    "beam":        (4, "beam overhead — duck",                  "⚠️"),
    "pipe":        (4, "pipe in path",                          "⚠️"),
    "construction":(4, "construction zone — be careful",        "🏗️"),
    "scaffold":    (4, "scaffolding overhead",                  "🏗️"),
    "dog":         (4, "dog nearby — approach carefully",       "🐕"),
    "animal":      (4, "animal nearby — be cautious",           "🐾"),
    "snake":       (4, "snake nearby — do not approach",        "🐍"),
    "person":      (4, "person directly ahead — slow down",     "🧍"),
    "people":      (4, "people ahead — slow down",              "👥"),
    "child":       (4, "child nearby — be extra careful",       "👶"),
    "baby":        (4, "baby nearby — be extra careful",        "👶"),
    "bicycle":     (4, "bicycle nearby",                        "🚲"),
    "bike":        (4, "bicycle nearby",                        "🚲"),
    "glass":       (4, "glass nearby — be careful",             "⚠️"),
    "broken":      (4, "broken object nearby",                  "⚠️"),
    "sharp":       (4, "sharp object nearby",                   "⚠️"),
    "debris":      (4, "debris on ground",                      "⚠️"),
    "rubble":      (4, "rubble on ground",                      "⚠️"),

    # PRIORITY 5 — LOW
    "chair":       (5, "chair in path",                         "🪑"),
    "stool":       (5, "stool in path",                         "🪑"),
    "table":       (5, "table ahead",                           "🪑"),
    "desk":        (5, "desk ahead",                            "🪑"),
    "bench":       (5, "bench ahead",                           "🪑"),
    "sofa":        (5, "sofa in path",                          "🛋️"),
    "couch":       (5, "couch in path",                         "🛋️"),
    "box":         (5, "box in path",                           "📦"),
    "crate":       (5, "crate in path",                         "📦"),
    "luggage":     (5, "luggage in path",                       "🧳"),
    "suitcase":    (5, "suitcase in path",                      "🧳"),
    "cord":        (5, "cord on ground — trip hazard",          "⚠️"),
    "cable":       (5, "cable on ground — trip hazard",         "⚠️"),
    "wire":        (5, "wire on ground — trip hazard",          "⚠️"),
    "hose":        (5, "hose on ground — trip hazard",          "⚠️"),
    "rope":        (5, "rope on ground — trip hazard",          "⚠️"),
    "mat":         (5, "mat on floor — edge risk",              "⚠️"),
    "rug":         (5, "rug on floor — edge risk",              "⚠️"),
    "carpet":      (5, "carpet edge — trip risk",               "⚠️"),
    "clutter":     (5, "clutter on floor",                      "⚠️"),
}


def check_for_hazards(image: Image.Image, scene_description: str = "") -> dict:
    """
    Scan the scene description for the highest-priority hazard keyword.
    No extra API call needed — uses the caption already generated.
    """
    if not scene_description:
        return {
            "hazard_detected": False,
            "hazard_type":     "",
            "hazard_emoji":    "",
            "hazard_priority": 99,
            "matched_keyword": "",
        }

    text = scene_description.lower()
    print(f"[Hazard scan] '{text[:80]}...'")

    best_priority = 99
    best_label    = ""
    best_emoji    = ""
    best_keyword  = ""

    for keyword, (priority, label, emoji) in HAZARD_KEYWORDS.items():
        if keyword in text and priority < best_priority:
            best_priority = priority
            best_label    = label
            best_emoji    = emoji
            best_keyword  = keyword

    if best_label:
        print(f"[Hazard] ⚠️  priority={best_priority} '{best_keyword}' → '{best_label}'")
    else:
        print("[Hazard] ✓ no hazard found")

    return {
        "hazard_detected": bool(best_label),
        "hazard_type":     best_label,
        "hazard_emoji":    best_emoji,
        "hazard_priority": best_priority,
        "matched_keyword": best_keyword,
    }
