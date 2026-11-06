"""Plugin TTS provider ``voxtral-realtime``: Voxtral 3 realtime streaming (PCM) + batch fallback.

Registered via ``ctx.register_tts_provider``; selected with ``tts.provider: voxtral-realtime``
(or pinned for the gateway streaming path with ``tts.streaming.provider: voxtral-realtime``).
``streams_pcm`` opts into the gateway's streaming-TTS consumer, so voice-call replies are
spoken sentence-by-sentence while the LLM is still generating.
"""

from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from agent.tts_provider import TTSProvider
from gateway.platforms._shared import platform_gate_env

from .settings import (
    DEFAULT_TTS_BATCH_MODEL,
    DEFAULT_TTS_REALTIME_MODEL,
    DEFAULT_TTS_SAMPLE_RATE,
    DEFAULT_TTS_VOICE,
    DEFAULT_TTS_VOICE_ID,
)
from .voxtral import RealtimeTTSStreamer, batch_speech, mistral_api_key

VALID_OUTPUT_FORMATS = frozenset({"mp3", "wav", "ogg", "opus", "flac", "pcm"})


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
            {"id": "en_paul_neutral", "display": "Paul (neutral, English)"},
            {"id": DEFAULT_TTS_VOICE_ID, "display": "Paul neutral (batch voice ID)"},
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
        voice_id = voice if voice and len(voice) == 36 and "-" in voice else (
            platform_gate_env("VOICE_CALL_TTS_VOICE_ID", DEFAULT_TTS_VOICE_ID) or DEFAULT_TTS_VOICE_ID
        )
        data = batch_speech(
            base_url=self._base_url(),
            model=model or platform_gate_env("VOICE_CALL_TTS_BATCH_MODEL", DEFAULT_TTS_BATCH_MODEL) or DEFAULT_TTS_BATCH_MODEL,
            api_key=mistral_api_key(),
            text=text,
            voice_id=voice_id,
            output_format=fmt,
            sample_rate=self.stream_sample_rate,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def stream(
        self, text: str, *, voice: Optional[str] = None, model: Optional[str] = None,
        format: str = "pcm", **extra: Any,
    ) -> Iterator[bytes]:
        """Stream int16 mono PCM at ``stream_sample_rate`` for one sentence of text."""
        if not text or not text.strip():
            return
        streamer = RealtimeTTSStreamer(
            base_url=self._base_url(),
            model=model or self.default_model(),
            api_key=mistral_api_key(),
            voice=voice or self.default_voice(),
            sample_rate=self.stream_sample_rate,
        )
        yield from streamer.stream(text)
