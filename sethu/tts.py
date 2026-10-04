"""Server-side text to speech, so read-aloud works on any device regardless of installed voices.

Uses the `edge-tts` package, which calls Microsoft's neural voices without an API key. That is
an unofficial route: fine for a prototype, but a deployment should switch to an official
service (Bhashini, Azure or Google Cloud TTS). The answer text is sent to that service.
"""

import re
from collections import OrderedDict

import edge_tts

# (script pattern, language code, voice). Order matters only for readability; scripts do not overlap.
SCRIPTS = [
    (re.compile(r"[ಀ-೿]"), "kn"),
    (re.compile(r"[ఀ-౿]"), "te"),
    (re.compile(r"[஀-௿]"), "ta"),
    (re.compile(r"[ഀ-ൿ]"), "ml"),
    (re.compile(r"[ঀ-৿]"), "bn"),
    (re.compile(r"[ऀ-ॿ]"), "hi"),  # Hindi and Marathi share Devanagari
]
VOICES = {
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "kn": "kn-IN-SapnaNeural",
    "te": "te-IN-ShrutiNeural",
    "ta": "ta-IN-PallaviNeural",
    "ml": "ml-IN-SobhanaNeural",
    "bn": "bn-IN-TanishaaNeural",
    "mr": "mr-IN-AarohiNeural",
}
MAX_CHARS = 3500
_cache: OrderedDict[tuple[str, str], bytes] = OrderedDict()
_CACHE_SIZE = 32


def clean_for_speech(text: str) -> str:
    """Drop citations, markdown and URLs, which sound like noise when read out."""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*#_`>|]", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    return re.sub(r"[ \t]+", " ", text).strip()


def detect_language(text: str, hint: str | None = None) -> str:
    """Language code from the text's script; `hint` ("mr") breaks the Hindi/Marathi tie."""
    for pattern, code in SCRIPTS:
        if pattern.search(text):
            return "mr" if code == "hi" and hint == "mr" else code
    return "en"


async def synthesize(text: str, hint: str | None = None) -> bytes:
    """MP3 audio of the text, spoken in the language it is written in."""
    text = clean_for_speech(text)[:MAX_CHARS]
    if not text:
        raise ValueError("nothing to read")
    voice = VOICES[detect_language(text, hint)]
    key = (voice, text)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]

    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, voice).stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    if not audio:
        raise RuntimeError("the speech service returned no audio")
    _cache[key] = bytes(audio)
    if len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return _cache[key]
