"""Voice-call settings: the ``platforms.voice-call:`` config.yaml block plus .env, read profile-scoped."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from gateway.platforms._shared import extra_or_secret, get_scoped_secret

DEFAULT_WEBHOOK_HOST = "127.0.0.1"
DEFAULT_WEBHOOK_PORT = 8081
DEFAULT_MISTRAL_BASE_URL = "https://api.mistral.ai"
DEFAULT_STT_MODEL = "voxtral-mini-transcribe-realtime-3-0"
DEFAULT_STT_DELAY_MS = 480
DEFAULT_TTS_REALTIME_MODEL = "voxtral-mini-tts-realtime-3"
DEFAULT_TTS_BATCH_MODEL = "voxtral-mini-tts-3-0"
DEFAULT_TTS_VOICE = "en_paul_neutral"
DEFAULT_TTS_VOICE_ID = "en_paul_neutral"
DEFAULT_TTS_SAMPLE_RATE = 24000
DEFAULT_MAX_CALL_SECONDS = 1800
PHONE_SAMPLE_RATE = 8000


def _csv(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple)):
        return [str(v).strip() for v in raw if str(v).strip()]
    return [p.strip() for p in str(raw or "").split(",") if p.strip()]


def _int(raw: Any, default: int) -> int:
    with contextlib.suppress(ValueError, TypeError):
        return int(raw)
    return default


def _bool(raw: Any) -> bool:
    return str(raw or "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class VoiceCallSettings:
    webhook_host: str
    webhook_port: int
    webhook_url: str
    insecure_no_signature: bool
    from_number: str
    allowed_numbers: frozenset
    allow_all: bool
    consent_text: str
    max_call_seconds: int
    mistral_base_url: str
    stt_model: str
    stt_delay_ms: int
    stt_language: str
    stt_context_bias: tuple
    turn_detection: dict
    tts_realtime_model: str
    tts_batch_model: str
    tts_voice: str
    tts_voice_id: str
    tts_sample_rate: int

    @classmethod
    def from_extra(cls, extra: dict) -> "VoiceCallSettings":
        extra = extra or {}

        def get(key: str, env: str, default: Any = "") -> Any:
            return extra_or_secret(extra, key, env, default)

        turn_detection_raw = extra_or_secret(extra, "turn_detection", "VOICE_CALL_TURN_DETECTION", "")
        turn_detection: dict = {}
        if isinstance(turn_detection_raw, dict):
            turn_detection = dict(turn_detection_raw)
        elif str(turn_detection_raw or "").strip().lower() in ("1", "true", "yes", "on"):
            turn_detection = {}
        return cls(
            webhook_host=str(get("webhook_host", "VOICE_CALL_WEBHOOK_HOST", DEFAULT_WEBHOOK_HOST) or DEFAULT_WEBHOOK_HOST),
            webhook_port=_int(get("webhook_port", "VOICE_CALL_WEBHOOK_PORT", str(DEFAULT_WEBHOOK_PORT)), DEFAULT_WEBHOOK_PORT),
            webhook_url=str(get("webhook_url", "VOICE_CALL_WEBHOOK_URL", "") or "").strip(),
            insecure_no_signature=_bool(get("insecure_no_signature", "VOICE_CALL_INSECURE_NO_SIGNATURE", "")),
            from_number=str(get("from_number", "TWILIO_PHONE_NUMBER", "") or "").strip(),
            allowed_numbers=frozenset(_csv(get("allowed_numbers", "VOICE_CALL_ALLOWED_NUMBERS", ""))),
            allow_all=_bool(get("allow_all", "VOICE_CALL_ALLOW_ALL_USERS", "")),
            consent_text=str(get("consent_text", "VOICE_CALL_CONSENT_TEXT", "") or "").strip(),
            max_call_seconds=_int(get("max_call_seconds", "VOICE_CALL_MAX_CALL_SECONDS", str(DEFAULT_MAX_CALL_SECONDS)), DEFAULT_MAX_CALL_SECONDS),
            mistral_base_url=str(get("mistral_base_url", "VOICE_CALL_MISTRAL_BASE_URL", DEFAULT_MISTRAL_BASE_URL) or DEFAULT_MISTRAL_BASE_URL).strip().rstrip("/"),
            stt_model=str(get("stt_model", "VOICE_CALL_STT_MODEL", DEFAULT_STT_MODEL) or DEFAULT_STT_MODEL),
            stt_delay_ms=_int(get("stt_delay_ms", "VOICE_CALL_STT_DELAY_MS", str(DEFAULT_STT_DELAY_MS)), DEFAULT_STT_DELAY_MS),
            stt_language=str(get("stt_language", "VOICE_CALL_STT_LANGUAGE", "") or "").strip(),
            stt_context_bias=tuple(_csv(get("stt_context_bias", "VOICE_CALL_STT_CONTEXT_BIAS", ""))),
            turn_detection=turn_detection,
            tts_realtime_model=str(get("tts_realtime_model", "VOICE_CALL_TTS_MODEL", DEFAULT_TTS_REALTIME_MODEL) or DEFAULT_TTS_REALTIME_MODEL),
            tts_batch_model=str(get("tts_batch_model", "VOICE_CALL_TTS_BATCH_MODEL", DEFAULT_TTS_BATCH_MODEL) or DEFAULT_TTS_BATCH_MODEL),
            tts_voice=str(get("tts_voice", "VOICE_CALL_TTS_VOICE", DEFAULT_TTS_VOICE) or DEFAULT_TTS_VOICE),
            tts_voice_id=str(get("tts_voice_id", "VOICE_CALL_TTS_VOICE_ID", DEFAULT_TTS_VOICE_ID) or DEFAULT_TTS_VOICE_ID),
            tts_sample_rate=_int(get("tts_sample_rate", "VOICE_CALL_TTS_SAMPLE_RATE", str(DEFAULT_TTS_SAMPLE_RATE)), DEFAULT_TTS_SAMPLE_RATE),
        )

    def mistral_api_key(self) -> str:
        return str(get_scoped_secret("MISTRAL_API_KEY", "") or "").strip()

    def caller_allowed(self, from_number: str) -> bool:
        if self.allow_all:
            return True
        if not self.allowed_numbers:
            return True
        return (from_number or "").strip() in self.allowed_numbers
