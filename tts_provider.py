"""Plugin TTS provider ``voxtral-realtime``: Voxtral 3 realtime streaming (PCM) + batch fallback.

Registered via ``ctx.register_tts_provider``; selected with ``tts.provider: voxtral-realtime``
(or pinned for the gateway streaming path with ``tts.streaming.provider: voxtral-realtime``).
``streams_pcm`` opts into the gateway's streaming-TTS consumer, so voice-call replies are
spoken sentence-by-sentence while the LLM is still generating. Synthesis runs over a POOL of
persistent realtime-TTS websockets (one per model/voice), so per-sentence time-to-first-audio
stays at the model's ~90 ms instead of paying a reconnect per sentence. Voices created in
Mistral Studio (clones, ``voice_<n>`` ids) are listed dynamically via GET /v1/audio/voices.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from agent.tts_provider import TTSProvider
from gateway.platforms._shared import platform_gate_env

logger = logging.getLogger("voxtral-voice-call")

from .settings import (
    DEFAULT_TTS_BATCH_MODEL,
    DEFAULT_TTS_REALTIME_MODEL,
    DEFAULT_TTS_SAMPLE_RATE,
    DEFAULT_TTS_VOICE,
)
from .voxtral import RealtimeTTSPool, RealtimeTTSSession, batch_speech, list_account_voices, mistral_api_key

VALID_OUTPUT_FORMATS = frozenset({"pcm", "wav", "mp3", "flac", "opus"})

# Voxtral 3 preset voices (ids + Studio-style names/languages) plus the Voxtral 2 emotion
# voices, which v3 models also accept. (id, label); v2 ids are their own label.
VOXTRAL3_VOICES = [
    ("en_paul_neutral", "en_paul_neutral (Paul, neutral en)"),
    ("en_paul_happy", "en_paul_happy (Paul, happy en)"),
    ("en_paul_confident", "en_paul_confident (Paul, confident en)"),
    ("gb_jane_neutral", "gb_jane_neutral (Jane, neutral en-GB)"),
    ("gb_oliver_neutral", "gb_oliver_neutral (Oliver, neutral en-GB)"),
    ("fr_marie_neutral", "fr_marie_neutral (Marie, neutral fr)"),
    ("voice_3150", "James (en-US)"),
    ("voice_447", "Margaret (en-US)"),
    ("voice_485", "Rob (en-US)"),
    ("voice_478", "Sheila (en-US)"),
    ("voice_1788", "David (en-GB)"),
    ("voice_591", "Fred (en-GB)"),
    ("voice_590", "Harriet (en-GB)"),
    ("voice_3147", "Howard (en-GB)"),
    ("voice_3149", "Hugo (en-GB)"),
    ("voice_1787", "Mary (en-GB)"),
    ("voice_1786", "Sky (en-GB)"),
    ("voice_483", "Elena (es)"),
    ("voice_3141", "Camille (fr)"),
    ("voice_589", "Lea (fr)"),
    ("voice_3137", "Manon (fr)"),
    ("voice_592", "Marie (fr)"),
    ("voice_3139", "Ananya (hi)"),
    ("voice_3153", "Himanshu (hi)"),
    ("voice_3146", "Olivia (hu)"),
    ("voice_482", "Lorenzo (it)"),
    ("voice_3133", "Haruto (ja)"),
    ("voice_3135", "Jin (ko)"),
    ("voice_3136", "Sung (ko)"),
    ("voice_3145", "Lotte (nl)"),
    ("voice_3134", "Lars (no)"),
    ("voice_3154", "Leon (pl)"),
    ("voice_480", "Gabriela (pt)"),
    ("voice_3155", "Irina (ro)"),
    ("voice_3132", "Tatiana (ru)"),
    ("voice_3138", "Oran (tr)"),
    ("voice_3140", "Taras (uk)"),
    ("voice_3144", "Huong (vi)"),
    ("voice_3142", "Mei (zh)"),
    ("voice_479", "Maryam (ar)"),
    ("voice_3143", "Jordi (ca)"),
    ("voice_481", "Heidi (de)"),
]

_VOICE_IDS = frozenset(vid for vid, _ in VOXTRAL3_VOICES)
_VOICE_LABELS = {label: vid for vid, label in VOXTRAL3_VOICES}


def resolve_voice_id(value: Optional[str]) -> str:
    """Resolve a configured voice (preset id, Studio-style label, or clone id) to the API id."""
    v = (value or "").strip()
    if not v:
        return DEFAULT_TTS_VOICE
    if v in _VOICE_IDS:
        return v
    return _VOICE_LABELS.get(v, v)

_POOLS: Dict[tuple, RealtimeTTSPool] = {}
_POOLS_LOCK = threading.Lock()

# (monotonic fetched_at, custom voices) — the picker calls list_voices on every paint, so cache.
_CUSTOM_VOICES_CACHE: list = [0.0, []]
_CUSTOM_VOICES_TTL = 60.0


def _custom_voices(base_url: str, api_key: str) -> list:
    import time as _time
    now = _time.monotonic()
    if _CUSTOM_VOICES_CACHE[1] and now - _CUSTOM_VOICES_CACHE[0] < _CUSTOM_VOICES_TTL:
        return _CUSTOM_VOICES_CACHE[1]
    try:
        account = list_account_voices(base_url=base_url, api_key=api_key)
        customs = [v for v in account if v["id"] not in _VOICE_IDS]
        _CUSTOM_VOICES_CACHE[0] = now
        _CUSTOM_VOICES_CACHE[1] = customs
        if customs:
            logger.info("[voxtral] %d custom/cloned voice(s) from Mistral account", len(customs))
        return customs
    except Exception as exc:
        logger.debug("[voxtral] account voice list failed (presets still shown): %s", exc)
        return _CUSTOM_VOICES_CACHE[1]


def _pool_for(key: tuple, base_url: str, api_key: str) -> RealtimeTTSPool:
    with _POOLS_LOCK:
        pool = _POOLS.get(key)
        if pool is None:
            model, voice, rate = key
            pool = RealtimeTTSPool(lambda: RealtimeTTSSession(
                base_url=base_url, model=model, api_key=api_key, voice=voice, sample_rate=rate,
            ))
            _POOLS[key] = pool
        return pool


class VoxtralRealtimeTTSProvider(TTSProvider):
    name = "voxtral-realtime"

    @property
    def streams_pcm(self) -> bool:
        return True

    @property
    def stream_sample_rate(self) -> int:
        with contextlib.suppress(ValueError):
            return int(platform_gate_env("VOICE_CALL_TTS_SAMPLE_RATE", str(DEFAULT_TTS_SAMPLE_RATE)))
        return DEFAULT_TTS_SAMPLE_RATE

    def is_available(self) -> bool:
        if not mistral_api_key():
            return False
        return importlib.util.find_spec("websockets") is not None

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {"id": DEFAULT_TTS_REALTIME_MODEL, "display": "Voxtral Mini TTS Realtime 3 (streaming)"},
            {"id": DEFAULT_TTS_BATCH_MODEL, "display": "Voxtral Mini TTS 3 (batch /v1/audio/speech)"},
        ]

    def default_model(self) -> Optional[str]:
        return platform_gate_env("VOICE_CALL_TTS_MODEL", DEFAULT_TTS_REALTIME_MODEL) or DEFAULT_TTS_REALTIME_MODEL

    def default_voice(self) -> Optional[str]:
        return resolve_voice_id(platform_gate_env("VOICE_CALL_TTS_VOICE", DEFAULT_TTS_VOICE) or DEFAULT_TTS_VOICE)

    def list_voices(self) -> List[Dict[str, Any]]:
        voices = [{"id": vid, "display": label} for vid, label in VOXTRAL3_VOICES]
        for custom in _custom_voices(self._base_url(), mistral_api_key()):
            voices.append({"id": custom["id"], "display": f"{custom['name']} (clone)"})
        return voices

    def get_setup_schema(self) -> dict:
        return {
            "name": self.display_name,
            "badge": "voxtral-3",
            "tag": "streaming",
            "env_vars": [{"key": "MISTRAL_API_KEY", "prompt": "Mistral API key", "url": "https://console.mistral.ai/"}],
        }

    def _base_url(self) -> str:
        return (platform_gate_env("VOICE_CALL_MISTRAL_BASE_URL", "https://api.mistral.ai") or "https://api.mistral.ai").rstrip("/")

    def synthesize(
        self, text: str, output_path: str, *, voice: Optional[str] = None, model: Optional[str] = None,
        speed: Optional[float] = None, format: str = "mp3", **extra: Any,
    ) -> str:
        fmt = format if format in VALID_OUTPUT_FORMATS else "mp3"
        path = Path(output_path)
        if fmt != format and path.suffix.lstrip(".") != fmt:
            path = path.with_suffix(f".{fmt}")
        voice_id = resolve_voice_id(voice or platform_gate_env("VOICE_CALL_TTS_VOICE_ID", "") or self.default_voice())
        data = batch_speech(
            base_url=self._base_url(),
            model=model or platform_gate_env("VOICE_CALL_TTS_BATCH_MODEL", DEFAULT_TTS_BATCH_MODEL) or DEFAULT_TTS_BATCH_MODEL,
            api_key=mistral_api_key(),
            text=text,
            voice_id=voice_id,
            response_format=fmt,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def stream(
        self, text: str, *, voice: Optional[str] = None, model: Optional[str] = None,
        format: str = "pcm", **extra: Any,
    ) -> Iterator[bytes]:
        """Stream int16 mono PCM at ``stream_sample_rate`` for one sentence over a pooled WS."""
        if not text or not text.strip():
            return
        key = (model or self.default_model(), resolve_voice_id(voice or self.default_voice()), self.stream_sample_rate)
        pool = _pool_for(key, self._base_url(), mistral_api_key())
        session = pool.acquire()
        logger.info("[voxtral] tts stream() sentence (%d chars) via pool idle=%d", len(text), len(pool._idle))
        try:
            yield from session.stream_sentence(text)
        except Exception as exc:
            logger.error("[voxtral] tts stream() failed: %s", exc, exc_info=True)
            raise
        finally:
            pool.release(session)
