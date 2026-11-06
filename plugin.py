"""Plugin entry point: registers the ``voice-call`` gateway platform and the ``voxtral-realtime`` TTS provider."""

from __future__ import annotations

from typing import Optional

PLATFORM_NAME = "voice-call"
TTS_PROVIDER_NAME = "voxtral-realtime"

_YAML_SPEC = (
    ("webhook_host", "VOICE_CALL_WEBHOOK_HOST", "str"),
    ("webhook_port", "VOICE_CALL_WEBHOOK_PORT", "str"),
    ("webhook_url", "VOICE_CALL_WEBHOOK_URL", "str"),
    ("insecure_no_signature", "VOICE_CALL_INSECURE_NO_SIGNATURE", "str"),
    ("from_number", "TWILIO_PHONE_NUMBER", "str"),
    ("allowed_numbers", "VOICE_CALL_ALLOWED_NUMBERS", "csv"),
    ("allow_all", "VOICE_CALL_ALLOW_ALL_USERS", "str"),
    ("consent_text", "VOICE_CALL_CONSENT_TEXT", "str"),
    ("max_call_seconds", "VOICE_CALL_MAX_CALL_SECONDS", "str"),
    ("mistral_base_url", "VOICE_CALL_MISTRAL_BASE_URL", "lower"),
    ("stt_model", "VOICE_CALL_STT_MODEL", "str"),
    ("stt_delay_ms", "VOICE_CALL_STT_DELAY_MS", "str"),
    ("stt_language", "VOICE_CALL_STT_LANGUAGE", "str"),
    ("stt_context_bias", "VOICE_CALL_STT_CONTEXT_BIAS", "csv"),
    ("tts_realtime_model", "VOICE_CALL_TTS_MODEL", "str"),
    ("tts_batch_model", "VOICE_CALL_TTS_BATCH_MODEL", "str"),
    ("tts_voice", "VOICE_CALL_TTS_VOICE", "str"),
    ("tts_voice_id", "VOICE_CALL_TTS_VOICE_ID", "str"),
    ("tts_sample_rate", "VOICE_CALL_TTS_SAMPLE_RATE", "str"),
)

_ENV_SEED_SPEC = (
    ("TWILIO_PHONE_NUMBER", "from_number", None),
    ("VOICE_CALL_WEBHOOK_HOST", "webhook_host", None),
    ("VOICE_CALL_WEBHOOK_PORT", "webhook_port", None),
    ("VOICE_CALL_WEBHOOK_URL", "webhook_url", None),
    ("VOICE_CALL_ALLOWED_NUMBERS", "allowed_numbers", None),
    ("VOICE_CALL_ALLOW_ALL_USERS", "allow_all", None),
    ("VOICE_CALL_CONSENT_TEXT", "consent_text", None),
    ("VOICE_CALL_MAX_CALL_SECONDS", "max_call_seconds", None),
    ("VOICE_CALL_MISTRAL_BASE_URL", "mistral_base_url", None),
    ("VOICE_CALL_STT_MODEL", "stt_model", None),
    ("VOICE_CALL_STT_DELAY_MS", "stt_delay_ms", None),
    ("VOICE_CALL_TTS_MODEL", "tts_realtime_model", None),
    ("VOICE_CALL_TTS_VOICE", "tts_voice", None),
    ("VOICE_CALL_TTS_VOICE_ID", "tts_voice_id", None),
    ("VOICE_CALL_TTS_SAMPLE_RATE", "tts_sample_rate", None),
)

_PLATFORM_HINT = (
    "You are on a live phone call (Twilio). The user's speech is transcribed and sent to you as "
    "voice messages; your replies are spoken aloud. Keep answers short, conversational, and free "
    "of markdown, lists, code, and URLs - they will be read out. If a transcript looks garbled, "
    "ask a brief clarifying question. Never claim an action succeeded before it actually did."
)


def _env_seed() -> dict:
    from gateway.platforms._shared import seed_extra_from_env
    return seed_extra_from_env(_ENV_SEED_SPEC, home_env="VOICE_CALL_HOME_CHANNEL")


def _apply_yaml(cfg: dict) -> Optional[dict]:
    from gateway.platforms._shared import apply_yaml_bridge
    return apply_yaml_bridge(cfg, _YAML_SPEC)


def register(ctx) -> None:
    from gateway.platforms.send_targets import StandaloneSendPolicy

    from .adapter import VoiceCallAdapter, check_requirements, is_connected
    from .tts_provider import VoxtralRealtimeTTSProvider

    ctx.register_platform(
        name=PLATFORM_NAME,
        label="Voice Calls (Twilio + Voxtral 3)",
        adapter_factory=VoiceCallAdapter,
        check_fn=check_requirements,
        is_connected=is_connected,
        required_env=["MISTRAL_API_KEY", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER"],
        install_hint="pip install aiohttp websockets",
        env_enable_credentials=("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"),
        env_enablement_fn=_env_seed,
        apply_yaml_config_fn=_apply_yaml,
        allowed_users_env="VOICE_CALL_ALLOWED_NUMBERS",
        allow_all_env="VOICE_CALL_ALLOW_ALL_USERS",
        allowlist_suppresses_pairing=True,
        standalone_send=StandaloneSendPolicy(text_direct=True),
        pii_safe=True,
        emoji="📞",
        max_message_length=0,
        platform_hint=_PLATFORM_HINT,
        display_tier="minimal",
        dashboard_description=(
            "Answer phone calls on your Twilio number and talk to Hermes live. Caller speech is "
            "transcribed by Voxtral Mini Transcribe Realtime 3 and replies are spoken with Voxtral "
            "Mini TTS Realtime 3 (streaming, with barge-in)."
        ),
        docs_url="https://www.twilio.com/docs/voice/twiml/stream",
        dashboard_env_vars=("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER", "MISTRAL_API_KEY"),
        dashboard_required_env=("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER", "MISTRAL_API_KEY"),
    )
    ctx.register_tts_provider(VoxtralRealtimeTTSProvider())
