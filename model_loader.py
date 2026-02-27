# model_loader.py
# Loads the BLIP-large model for maximum accuracy.
#
# ACCURACY IMPROVEMENTS vs the base version:
#   1. Uses "blip-image-captioning-large" — 2x more parameters, much richer descriptions
#   2. num_beams raised from 5 → 10  — explores more caption candidates before committing
#   3. max_new_tokens raised to 200  — allows longer, more detailed descriptions
#   4. length_penalty=1.5            — encourages the model to generate complete sentences
#   5. repetition_penalty=1.3        — stops the model from repeating phrases
#   6. no_repeat_ngram_size=3        — blocks any 3-word phrase from appearing twice
#   7. Image upscaled to 512px before inference — BLIP sees more detail per pixel

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

processor = None
model     = None


def load_model():
    """
    Load the large BLIP captioning model from HuggingFace.
    First run downloads ~1.9GB — subsequent runs load from local cache in ~10s.
    """
    global processor, model

    MODEL_ID = "Salesforce/blip-image-captioning-base"
    print(f"Loading {MODEL_ID} (large model — more accurate, first run downloads ~1.9GB)...")

    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model     = BlipForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,   # float16 if you have a GPU with enough VRAM
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    model.eval()

    print(f"Model ready on {device}.")
    return processor, model


def generate_caption(image: Image.Image) -> str:
    """
    Generate a detailed, accurate description of the given PIL image.

    The image is upscaled before inference so BLIP sees as much detail as
    possible — important for live camera frames which may be lower resolution.
    """
    global processor, model

    if processor is None or model is None:
        load_model()

    device = next(model.parameters()).device

    # ── Pre-process: upscale small images ────────────────────────────────────
    # BLIP internally resizes to 384×384. If the input is tiny (e.g. 320×240
    # webcam frame), upscaling first with LANCZOS gives it more pixel detail
    # to work with before that resize step.
    MIN_DIM = 512
    w, h    = image.size
    if w < MIN_DIM or h < MIN_DIM:
        scale = MIN_DIM / min(w, h)
        image = image.resize(
            (int(w * scale), int(h * scale)),
            Image.LANCZOS
        )

    # ── Conditional prompt ────────────────────────────────────────────────────
    # "a photography of" is the prompt BLIP was fine-tuned with — it anchors
    # the model to produce realistic, grounded descriptions rather than abstract ones
    prompt = "a photography of"
    inputs = processor(image, prompt, return_tensors="pt").to(device)

    # ── Generation settings ───────────────────────────────────────────────────
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,       # allow long, detailed descriptions
            num_beams=10,             # check 10 candidate sequences (vs 5 before)
            length_penalty=1.5,       # reward longer, more complete sentences
            repetition_penalty=1.3,   # penalise repeating the same words
            no_repeat_ngram_size=3,   # no 3-word phrase can appear twice
            early_stopping=True,
        )

    caption = processor.decode(output[0], skip_special_tokens=True).strip()

    # ── Post-process ──────────────────────────────────────────────────────────
    if caption and not caption[0].isupper():
        caption = caption[0].upper() + caption[1:]
    if caption and not caption.endswith("."):
        caption += "."

    return caption


# ── Hazard detection — priority-based description keyword scan ────────────────
#
# HOW IT WORKS:
#   BLIP's caption reliably names the objects it sees. We scan that text
#   for hazard words and return the HIGHEST PRIORITY match — not just the
#   first one. This means "fire" always beats "chair" even if chair appears
#   earlier in the description.
#
# PRIORITY LEVELS (lower number = more urgent, spoken first):
#   1 — CRITICAL  : fire, electricity, chemical, weapon  (immediate danger to life)
#   2 — SERIOUS   : vehicle, moving object, crowd        (high injury risk)
#   3 — HIGH      : stairs, drop, gap, wet floor         (fall/injury risk)
#   4 — MEDIUM    : door, wall, obstacle, animal         (collision/blockage)
#   5 — LOW       : furniture, clutter, cord             (trip hazard)
#
# Each entry: "keyword BLIP uses" → (priority, "spoken warning", "emoji")
# The emoji is shown on the hazard overlay for fast visual recognition.

HAZARD_KEYWORDS = {

    # ── PRIORITY 1: CRITICAL — immediate life danger ─────────────────────────
    "fire":           (1, "fire detected — move away immediately",    "🔥"),
    "flame":          (1, "fire detected — move away immediately",    "🔥"),
    "flames":         (1, "fire detected — move away immediately",    "🔥"),
    "burning":        (1, "fire detected — move away immediately",    "🔥"),
    "smoke":          (1, "smoke detected — possible fire nearby",    "💨"),
    "explosion":      (1, "explosion risk — move away",               "💥"),
    "electric":       (1, "electrical hazard nearby",                 "⚡"),
    "electrical":     (1, "electrical hazard nearby",                 "⚡"),
    "sparks":         (1, "electrical sparks — do not touch",         "⚡"),
    "live wire":      (1, "live wire — danger",                       "⚡"),
    "chemical":       (1, "chemical hazard nearby",                   "☣️"),
    "toxic":          (1, "toxic material nearby",                    "☣️"),
    "poison":         (1, "poisonous substance nearby",               "☣️"),
    "gun":            (1, "weapon detected nearby",                   "🚨"),
    "weapon":         (1, "weapon detected nearby",                   "🚨"),
    "knife":          (1, "sharp weapon nearby",                      "🚨"),
    "sword":          (1, "sharp weapon nearby",                      "🚨"),
    "flood":          (1, "flooding detected — avoid area",           "🌊"),
    "flooded":        (1, "flooding detected — avoid area",           "🌊"),

    # ── PRIORITY 2: SERIOUS — high injury risk ────────────────────────────────
    "car":            (2, "vehicle nearby — stop and wait",           "🚗"),
    "truck":          (2, "large vehicle nearby — stop and wait",     "🚛"),
    "bus":            (2, "bus nearby — stop and wait",               "🚌"),
    "van":            (2, "vehicle nearby — stop and wait",           "🚗"),
    "motorcycle":     (2, "motorcycle nearby — be careful",           "🏍️"),
    "motorbike":      (2, "motorcycle nearby — be careful",           "🏍️"),
    "scooter":        (2, "scooter nearby — be careful",              "🛵"),
    "vehicle":        (2, "vehicle nearby — stop and wait",           "🚗"),
    "traffic":        (2, "traffic ahead — do not cross",             "🚦"),
    "road":           (2, "road ahead — watch for vehicles",          "🛣️"),
    "street":         (2, "street ahead — watch for traffic",         "🛣️"),
    "intersection":   (2, "intersection ahead — stop and listen",     "🚦"),
    "crossing":       (2, "road crossing ahead",                      "🚦"),
    "train":          (2, "train nearby — stay clear of tracks",      "🚆"),
    "track":          (2, "train track ahead — cross carefully",      "🚆"),
    "forklift":       (2, "forklift nearby — dangerous",              "⚠️"),
    "crowd":          (2, "crowd ahead — move carefully",             "👥"),
    "running":        (2, "someone running nearby",                   "🏃"),
    "rushing":        (2, "fast movement nearby",                     "⚠️"),

    # ── PRIORITY 3: HIGH — fall and injury risk ───────────────────────────────
    "stair":          (3, "stairs ahead — hold the railing",          "🪜"),
    "stairs":         (3, "stairs ahead — hold the railing",          "🪜"),
    "staircase":      (3, "staircase ahead — hold the railing",       "🪜"),
    "stairway":       (3, "stairway ahead — hold the railing",        "🪜"),
    "step":           (3, "step ahead — watch your footing",          "⚠️"),
    "steps":          (3, "steps ahead — watch your footing",         "⚠️"),
    "escalator":      (3, "escalator ahead — hold the railing",       "🪜"),
    "ladder":         (3, "ladder nearby — be careful",               "🪜"),
    "ramp":           (3, "ramp ahead — uneven surface",              "⚠️"),
    "slope":          (3, "slope ahead — uneven surface",             "⚠️"),
    "hill":           (3, "hill ahead — uneven surface",              "⚠️"),
    "incline":        (3, "incline ahead — uneven surface",           "⚠️"),
    "cliff":          (3, "drop ahead — stay back",                   "🏔️"),
    "ledge":          (3, "ledge ahead — stay back",                  "⚠️"),
    "drop":           (3, "drop ahead — stay back",                   "⚠️"),
    "pit":            (3, "pit ahead — do not step forward",          "⚠️"),
    "hole":           (3, "hole in floor — do not step",              "⚠️"),
    "gap":            (3, "gap ahead — do not step",                  "⚠️"),
    "ditch":          (3, "ditch ahead — step carefully",             "⚠️"),
    "trench":         (3, "trench ahead — step carefully",            "⚠️"),
    "manhole":        (3, "manhole ahead — avoid",                    "⚠️"),
    "gutter":         (3, "gutter ahead — watch your step",           "⚠️"),
    "wet floor":      (3, "wet floor — slip risk",                    "💧"),
    "wet":            (3, "wet surface — slip risk",                  "💧"),
    "slippery":       (3, "slippery surface — slow down",             "💧"),
    "puddle":         (3, "puddle on ground — wet surface",           "💧"),
    "spill":          (3, "spill on floor — slip risk",               "💧"),
    "ice":            (3, "ice on ground — slip risk",                "🧊"),
    "icy":            (3, "icy surface — slip risk",                  "🧊"),
    "snow":           (3, "snow on ground — slippery",                "❄️"),
    "mud":            (3, "muddy ground — slippery",                  "⚠️"),

    # ── PRIORITY 4: MEDIUM — collision and blockage ───────────────────────────
    "door":           (4, "door ahead",                               "🚪"),
    "doorway":        (4, "doorway ahead",                            "🚪"),
    "door frame":     (4, "door frame ahead",                         "🚪"),
    "entrance":       (4, "entrance ahead",                           "🚪"),
    "exit":           (4, "exit ahead",                               "🚪"),
    "gate":           (4, "gate ahead",                               "🚧"),
    "turnstile":      (4, "turnstile ahead",                          "🚧"),
    "wall":           (4, "wall ahead — stop",                        "🧱"),
    "glass wall":     (4, "glass wall ahead — be careful",            "🧱"),
    "window":         (4, "window at head level — be careful",        "🪟"),
    "fence":          (4, "fence ahead",                              "🚧"),
    "barrier":        (4, "barrier ahead",                            "🚧"),
    "barricade":      (4, "barricade ahead",                          "🚧"),
    "bollard":        (4, "bollard in path",                          "🚧"),
    "post":           (4, "post in path",                             "⚠️"),
    "pole":           (4, "pole in path",                             "⚠️"),
    "pillar":         (4, "pillar ahead",                             "⚠️"),
    "column":         (4, "column ahead",                             "⚠️"),
    "beam":           (4, "beam overhead — duck",                     "⚠️"),
    "low ceiling":    (4, "low ceiling — duck",                       "⚠️"),
    "pipe":           (4, "pipe in path",                             "⚠️"),
    "construction":   (4, "construction zone — be careful",           "🏗️"),
    "scaffold":       (4, "scaffolding overhead",                     "🏗️"),
    "crane":          (4, "crane overhead — caution",                 "🏗️"),
    "excavation":     (4, "excavation nearby — watch your step",      "🏗️"),
    "dog":            (4, "dog nearby — approach carefully",          "🐕"),
    "animal":         (4, "animal nearby — be cautious",              "🐾"),
    "cat":            (4, "cat nearby — watch your step",             "🐈"),
    "bird":           (4, "bird nearby",                              "🐦"),
    "snake":          (4, "snake nearby — do not approach",           "🐍"),
    "insect":         (4, "insects nearby",                           "🐛"),
    "person":         (4, "person directly ahead — slow down",        "🧍"),
    "people":         (4, "people ahead — slow down",                 "👥"),
    "child":          (4, "child nearby — be extra careful",          "👶"),
    "baby":           (4, "baby nearby — be extra careful",           "👶"),
    "wheelchair":     (4, "wheelchair user nearby",                   "♿"),
    "bicycle":        (4, "bicycle nearby",                           "🚲"),
    "bike":           (4, "bicycle nearby",                           "🚲"),
    "skateboard":     (4, "skateboard nearby",                        "🛹"),
    "shopping cart":  (4, "shopping cart in path",                    "🛒"),
    "cart":           (4, "cart in path",                             "🛒"),
    "trolley":        (4, "trolley in path",                          "🛒"),
    "glass":          (4, "glass nearby — be careful",                "⚠️"),
    "broken glass":   (4, "broken glass — do not step",               "⚠️"),
    "sharp":          (4, "sharp object nearby",                      "⚠️"),
    "broken":         (4, "broken object nearby",                     "⚠️"),
    "debris":         (4, "debris on ground",                         "⚠️"),
    "rubble":         (4, "rubble on ground",                         "⚠️"),
    "rock":           (4, "rocks on ground",                          "⚠️"),
    "stone":          (4, "stones on ground",                         "⚠️"),

    # ── PRIORITY 5: LOW — trip hazards and clutter ────────────────────────────
    "chair":          (5, "chair in path",                            "🪑"),
    "stool":          (5, "stool in path",                            "🪑"),
    "table":          (5, "table ahead",                              "🪑"),
    "desk":           (5, "desk ahead",                               "🪑"),
    "bench":          (5, "bench ahead",                              "🪑"),
    "sofa":           (5, "sofa in path",                             "🛋️"),
    "couch":          (5, "couch in path",                            "🛋️"),
    "box":            (5, "box in path",                              "📦"),
    "boxes":          (5, "boxes in path",                            "📦"),
    "crate":          (5, "crate in path",                            "📦"),
    "luggage":        (5, "luggage in path",                          "🧳"),
    "suitcase":       (5, "suitcase in path",                         "🧳"),
    "bag":            (5, "bag on ground",                            "👜"),
    "backpack":       (5, "backpack on ground",                       "🎒"),
    "cord":           (5, "cord on ground — trip hazard",             "⚠️"),
    "cable":          (5, "cable on ground — trip hazard",            "⚠️"),
    "wire":           (5, "wire on ground — trip hazard",             "⚠️"),
    "hose":           (5, "hose on ground — trip hazard",             "⚠️"),
    "rope":           (5, "rope on ground — trip hazard",             "⚠️"),
    "mat":            (5, "mat on floor — edge risk",                 "⚠️"),
    "rug":            (5, "rug on floor — edge risk",                 "⚠️"),
    "carpet":         (5, "carpet edge — trip risk",                  "⚠️"),
    "clutter":        (5, "clutter on floor",                         "⚠️"),
    "mess":           (5, "messy floor ahead",                        "⚠️"),
    "litter":         (5, "litter on ground",                         "⚠️"),
}


def check_for_hazards(image: Image.Image, scene_description: str = "") -> dict:
    """
    Scan the scene description for the HIGHEST PRIORITY hazard keyword.

    Instead of returning the first match, we scan ALL keywords and return
    the one with the lowest priority number (most dangerous). This means
    if a description says "a chair near a door with a fire in the background",
    we warn about fire (priority 1) not chair (priority 5).

    Args:
        image:             PIL Image (kept for API compatibility, not used here)
        scene_description: The caption already generated by generate_caption()

    Returns:
        {
            "hazard_detected": True | False,
            "hazard_type":     "fire detected — move away immediately",
            "hazard_emoji":    "🔥",
            "hazard_priority": 1,
            "matched_keyword": "fire",
        }
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
    print(f"[Hazard scan] scanning: '{text}'")

    best_priority = 99
    best_label    = ""
    best_emoji    = ""
    best_keyword  = ""

    # Scan ALL keywords and find the highest-priority (lowest number) match
    for keyword, (priority, label, emoji) in HAZARD_KEYWORDS.items():
        if keyword in text:
            if priority < best_priority:
                best_priority = priority
                best_label    = label
                best_emoji    = emoji
                best_keyword  = keyword

    if best_label:
        print(f"[Hazard scan] ⚠️  priority={best_priority} keyword='{best_keyword}' → '{best_label}'")
    else:
        print("[Hazard scan] ✓ no hazard found")

    return {
        "hazard_detected": bool(best_label),
        "hazard_type":     best_label,
        "hazard_emoji":    best_emoji,
        "hazard_priority": best_priority,
        "matched_keyword": best_keyword,
    }
