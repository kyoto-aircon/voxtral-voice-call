"""Plugin TTS provider ``voxtral-realtime``: Voxtral 3 realtime streaming (PCM) + batch fallback.

Registered via ``ctx.register_tts_provider``; selected with ``tts.provider: voxtral-realtime``
(or pinned for the gateway streaming path with ``tts.streaming.provider: voxtral-realtime``).
``streams_pcm`` opts into the gateway's streaming-TTS consumer, so voice-call replies are
spoken sentence-by-sentence while the LLM is still generating. Synthesis runs over a POOL of
persistent realtime-TTS websockets (one per model/voice), so per-sentence time-to-first-audio
stays at the model's ~90 ms instead of paying a reconnect per sentence.
"""

from __future__ import annotations

import contextlib
import importlib.util
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from agent.tts_provider import TTSProvider
from gateway.platforms._shared import platform_gate_env

from .settings import (
    DEFAULT_TTS_BATCH_MODEL,
    DEFAULT_TTS_REALTIME_MODEL,
    DEFAULT_TTS_SAMPLE_RATE,
    DEFAULT_TTS_VOICE,
)
from .voxtral import RealtimeTTSPool, RealtimeTTSSession, batch_speech, mistral_api_key

VALID_OUTPUT_FORMATS = frozenset({"pcm", "wav", "mp3", "flac", "opus"})

_POOLS: Dict[tuple, RealtimeTTSPool] = {}
_POOLS_LOCK = threading.Lock()


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
        return platform_gate_env("VOICE_CALL_TTS_VOICE", DEFAULT_TTS_VOICE) or DEFAULT_TTS_VOICE

    def list_voices(self) -> List[Dict[str, Any]]:
        return [
            {"id": DEFAULT_TTS_VOICE, "display": "Paul (neutral, English) - Voxtral 3 preset"},
        ]

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
        voice_id = voice or platform_gate_env("VOICE_CALL_TTS_VOICE_ID", "") or self.default_voice()
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
        key = (model or self.default_model(), voice or self.default_voice(), self.stream_sample_rate)
        pool = _pool_for(key, self._base_url(), mistral_api_key())
        session = pool.acquire()
        try:
            yield from session.stream_sentence(text)
        finally:
            pool.release(session)
