"""Plugin STT provider ``voxtral-realtime``: Voxtral 3 realtime streaming + batch transcription.

Registered via ``ctx.register_transcription_provider``; selected with ``stt.provider: voxtral-realtime``.
``streaming_capable`` + ``open_stream_session`` opt into ``stt.streaming: true`` live sessions
(16 kHz mono s16le), which powers Desktop/CLI chained voice mode and any voice surface that
feeds live PCM. The file-based ``transcribe`` serves voice-note and Discord-voice utterances.
"""

from __future__ import annotations

import contextlib
import importlib.util
from typing import Any, Dict, List, Optional

from agent.transcription_provider import TranscriptionProvider, TranscriptionStreamSession
from gateway.platforms._shared import platform_gate_env

from .settings import DEFAULT_STT_DELAY_MS, DEFAULT_STT_MODEL
from .voxtral import RealtimeSTTSyncSession, batch_transcribe, mistral_api_key

STREAM_INPUT_RATE = 16000


class _VoxtralStreamSession(TranscriptionStreamSession):
    """Adapts the blocking websocket session to the TranscriptionStreamSession contract."""

    def __init__(self, inner: RealtimeSTTSyncSession) -> None:
        self._inner = inner

    def push_audio(self, chunk: bytes) -> None:
        self._inner.push_audio(chunk)

    def partial_transcript(self) -> str:
        return self._inner.partial_transcript()

    def finalize(self) -> Dict[str, Any]:
        try:
            return self._inner.finalize()
        except Exception as exc:
            return {"success": False, "transcript": "", "provider": self._inner.provider, "error": str(exc)}
        finally:
            self._inner.close()

    def close(self) -> None:
        self._inner.close()


class VoxtralRealtimeSTTProvider(TranscriptionProvider):
    name = "voxtral-realtime"

    @property
    def streaming_capable(self) -> bool:
        return True

    def is_available(self) -> bool:
        if not mistral_api_key():
            return False
        return importlib.util.find_spec("websockets") is not None

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {"id": DEFAULT_STT_MODEL, "display": "Voxtral Mini Transcribe Realtime 3 (streaming)"},
            {"id": "voxtral-mini-transcribe-3-0", "display": "Voxtral Mini Transcribe 3 (batch)"},
            {"id": "voxtral-small-transcribe-3-0", "display": "Voxtral Small Transcribe 3 (batch, most accurate)"},
        ]

    def default_model(self) -> Optional[str]:
        return platform_gate_env("VOICE_CALL_STT_MODEL", DEFAULT_STT_MODEL) or DEFAULT_STT_MODEL

    def _base_url(self) -> str:
        return (platform_gate_env("VOICE_CALL_MISTRAL_BASE_URL", "https://api.mistral.ai") or "https://api.mistral.ai").rstrip("/")

    def _context_bias(self, extra: Dict[str, Any]) -> tuple:
        bias = [p.strip() for p in platform_gate_env("VOICE_CALL_STT_CONTEXT_BIAS", "").split(",") if p.strip()]
        prompt = str(extra.get("prompt") or "").strip()
        if prompt:
            bias.extend(p.strip() for p in prompt.split(",") if p.strip())
        return tuple(bias)

    def transcribe(self, file_path: str, *, model: Optional[str] = None, language: Optional[str] = None,
                   **extra: Any) -> Dict[str, Any]:
        return batch_transcribe(
            base_url=self._base_url(),
            model=model or platform_gate_env("VOICE_CALL_STT_BATCH_MODEL", "voxtral-mini-transcribe-3-0") or "voxtral-mini-transcribe-3-0",
            api_key=mistral_api_key(),
            file_path=file_path,
            provider=self.name,
            language=language or platform_gate_env("VOICE_CALL_STT_LANGUAGE", ""),
            context_bias=self._context_bias(extra),
        )

    def open_stream_session(self, *, language: Optional[str] = None, prompt: Optional[str] = None) -> TranscriptionStreamSession:
        extra: Dict[str, Any] = {"prompt": prompt} if prompt else {}
        inner = RealtimeSTTSyncSession(
            base_url=self._base_url(),
            model=self.default_model(),
            api_key=mistral_api_key(),
            provider=self.name,
            sample_rate=STREAM_INPUT_RATE,
            delay_ms=_delay_ms(),
            language=language or platform_gate_env("VOICE_CALL_STT_LANGUAGE", ""),
            context_bias=self._context_bias(extra),
        )
        inner.open()
        return _VoxtralStreamSession(inner)

    def get_setup_schema(self) -> dict:
        return {
            "name": self.display_name,
            "badge": "voxtral-3",
            "tag": "streaming",
            "env_vars": [{"key": "MISTRAL_API_KEY", "prompt": "Mistral API key", "url": "https://console.mistral.ai/"}],
        }


def _delay_ms() -> int:
    with contextlib.suppress(ValueError):
        return int(platform_gate_env("VOICE_CALL_STT_DELAY_MS", str(DEFAULT_STT_DELAY_MS)))
    return DEFAULT_STT_DELAY_MS
