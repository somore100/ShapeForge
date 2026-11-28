"""Rule-based prompt cleanup for Shap-E. No AI, no network, instant.

Shap-E reads a short, concrete description of ONE object. It ignores render
tags (4k, masterpiece), has no background or lighting, and loses focus on long
prompts. These rules strip that noise and tidy the wording. They are a safe
baseline, not magic: compare results with a batch before trusting them.
"""

import re

RULES_LABEL = "Rules (no AI)"
MAX_WORDS = 24

STYLES = {
    "None": "",
    "Low-poly": "low-poly, flat colors",
    "Cartoon": "stylized cartoon look, bold simple shapes",
    "Realistic": "realistic proportions and materials",
    "Game asset": "clean game-ready asset",
    "Toy": "chunky toy-like, smooth rounded shapes",
}

# Phrases that mean nothing to a text-to-3D model.
_NOISE = (
    "highly detailed", "ultra detailed", "extremely detailed", "intricate details",
    "high quality", "high-quality", "best quality", "masterpiece",
    "award winning", "award-winning", "trending on artstation", "artstation",
    "octane render", "unreal engine", "8k", "4k", "uhd", "hd",
    "studio lighting", "dramatic lighting", "cinematic lighting",
    "soft lighting", "volumetric lighting", "depth of field", "bokeh",
)
_NOISE_RE = re.compile(
    r"(?<![\w-])(?:" + "|".join(re.escape(p) for p in _NOISE) + r")(?![\w-])", re.I)
# "on a white background", "plain background", "isolated on white" ...
_BACKGROUND_RE = re.compile(
    r"(?:\b(?:on|against|with|in|over)\s+)?(?:an?\s+)?"
    r"(?:(?:plain|simple|white|black|gr[ae]y|transparent|empty|neutral)\s+)+"
    r"background\b|\bisolated(?:\s+on\s+white)?\b|\bin the background\b", re.I)
_FILLER = {
    "very", "really", "extremely", "super", "quite", "amazing", "stunning",
    "awesome", "incredible", "gorgeous", "beautiful", "nice", "cool",
}
_DETERMINERS = {
    "a", "an", "the", "some", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "several", "many", "pair", "set",
    "group", "stack", "bunch", "couple", "dozen", "one",
}
# Bare nouns whose models often lose the parts that make them recognisable.
_PART_HINTS = {
    "with four wheels": {"car", "truck", "van", "bus", "jeep", "taxi", "pickup",
                         "sedan", "suv", "tractor", "wagon", "limousine"},
    "with two wheels": {"motorcycle", "motorbike", "bicycle", "bike", "scooter"},
    "with four legs": {"chair", "table", "desk"},
}
_PART_WORDS = ("wheel", "tire", "tyre", "leg", "wing", "rotor")
_QUOTES = "\"'`\u201c\u201d\u2018\u2019"
_A_BEFORE_VOWEL = ("uni", "use", "usu", "uti", "uri", "ute", "ubi", "eu", "one", "once")
_AN_BEFORE_H = ("hour", "honest", "honor", "honour", "heir")


def _article(word):
    low = word.lower()
    if low.startswith(_AN_BEFORE_H):
        return "an"
    if low[:1] in "aeiou" and not low.startswith(_A_BEFORE_VOWEL):
        return "an"
    return "a"


def _needs_article(first):
    word = re.sub(r"[^\w-]", "", first.lower())
    if not word or word in _DETERMINERS or word[0].isdigit():
        return False
    # Probably plural or a mass noun ("sneakers", "dice"): leave it alone.
    return not (len(word) > 3 and word.endswith("s") and not word.endswith("ss"))


def _add_parts(text):
    """Short prompts only: 'car' -> 'car with four wheels'."""
    words = [re.sub(r"[^\w]", "", w).lower() for w in text.split()]
    if not words or len(words) > 4 or any(
            part in w for w in words for part in _PART_WORDS):
        return text
    for hint, nouns in _PART_HINTS.items():
        if words[-1] in nouns:
            return f"{text} {hint}"
    return text


def _trim(text):
    words = text.split()
    if len(words) <= MAX_WORDS:
        return text
    head = " ".join(words[:MAX_WORDS])
    cut = max(head.rfind(","), head.rfind(" with "), head.rfind(" and "))
    if cut > len(head) // 2:  # cut at a natural break if it keeps most of it
        head = head[:cut]
    return head.rstrip(" ,;")


def rule_enhance(text, style="None"):
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) >= 2 and text[0] in _QUOTES and text[-1] in _QUOTES:
        text = text[1:-1].strip()

    text = _NOISE_RE.sub(" ", text)
    text = _BACKGROUND_RE.sub(" ", text)
    words = [w for w in text.split()
             if re.sub(r"[^\w]", "", w).lower() not in _FILLER]
    deduped = []
    for word in words:  # "red red cube" -> "red cube"
        if not deduped or word.lower().strip(",.;") != deduped[-1].lower().strip(",.;"):
            deduped.append(word)
    text = " ".join(deduped)
    text = re.sub(r"\s+([,.;])", r"\1", text)
    text = re.sub(r"([,;])(\s*[,;])+", r"\1", text)
    text = re.sub(r"^[\s,;.]+", "", text)
    text = _trim(text).rstrip(" .,;")
    text = _add_parts(text)

    if text and _needs_article(text.split()[0]):
        text = f"{_article(text.split()[0])} {text}"
    if text:
        text = text[0].upper() + text[1:]
    hint = STYLES.get(style, "")
    if text and hint and hint.lower() not in text.lower():
        text = f"{text}, {hint}"
    return text
