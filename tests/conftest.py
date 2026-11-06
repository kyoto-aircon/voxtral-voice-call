"""Stub the Hermes core modules the plugin imports, so tests run without a hermes install."""

from __future__ import annotations

import dataclasses
import enum
import importlib.util
import os
import sys
import types
from pathlib import Path

_SECRETS: dict = {}


def get_scoped_secret(name, default=None, **kwargs):
    return _SECRETS.get(name, default)


def platform_gate_env(name, default=""):
    return (os.environ.get(name) or default).strip()


def extra_or_secret(extra, key, env, default="", **kwargs):
    val = os.environ.get(env)
    if val is not None and str(val).strip() != "":
        return val
    if extra and key in extra and str(extra[key]).strip() != "":
        return extra[key]
    return default


def seed_extra_from_env(spec, home_env=None, home_default=""):
    seed = {}
    for env, key, conv in spec:
        raw = str(get_scoped_secret(env, "") or "").strip()
        if not raw:
            continue
        seed[key] = conv(raw) if conv else raw
    return seed


def apply_yaml_bridge(spec):
    return lambda cfg: None


def redact_phone(number):
    s = str(number or "")
    return s[:3] + "***" + s[-2:] if len(s) > 5 else "***"


def _mod(name, **attrs):
    module = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(module, k, v)
    sys.modules[name] = module
    return module


class _Platform(str):
    pass


@dataclasses.dataclass
class _PlatformConfig:
    enabled: bool = False
    extra: dict = dataclasses.field(default_factory=dict)


class _SendResult:
    def __init__(self, success=True, error=None, message_id=None):
        self.success = success
        self.error = error
        self.message_id = message_id


class _BasePlatformAdapter:
    serves_profile_prefix = False

    def __init__(self, config, platform):
        self.config = config
        self.platform = platform
        self._background_tasks = set()
        self.dispatched = []

    def _set_fatal_error(self, code, message, *, retryable=False):
        self.fatal = (code, message)

    def _wire_plugin_handlers(self, native=None):
        pass

    def build_source(self, chat_id, chat_name=None, chat_type="dm", user_id=None, user_name=None, message_id=None, **kw):
        return types.SimpleNamespace(platform=self.platform, chat_id=str(chat_id), chat_name=chat_name,
                                     chat_type=chat_type, user_id=user_id, user_name=user_name, message_id=message_id)

    async def handle_message(self, event):
        self.dispatched.append(event)


@dataclasses.dataclass
class _AudioFormat:
    sample_rate: int = 24000
    channels: int = 1
    sample_width: int = 2


@dataclasses.dataclass
class _StreamingTTSHandle:
    chat_id: str = ""
    audio_format: _AudioFormat = dataclasses.field(default_factory=_AudioFormat)
    audible: bool = False
    aborted: bool = False


class _MessageType(enum.Enum):
    TEXT = "text"
    AUDIO = "audio"
    VOICE = "voice"


@dataclasses.dataclass
class _MessageEvent:
    text: str
    message_type: _MessageType = _MessageType.TEXT
    source: object = None
    raw_message: object = None
    message_id: object = None
    metadata: dict = dataclasses.field(default_factory=dict)


class _TTSProvider:
    pass


class _TranscriptionProvider:
    pass


class _TranscriptionStreamSession:
    def push_audio(self, chunk):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

    def partial_transcript(self):
        return ""


class _StandaloneSendPolicy:
    def __init__(self, text_direct=False):
        self.text_direct = text_direct


def install_stubs():
    _mod("gateway")
    _mod("gateway.platforms")
    _shared = _mod("gateway.platforms._shared", get_scoped_secret=get_scoped_secret, platform_gate_env=platform_gate_env,
                   extra_or_secret=extra_or_secret, seed_extra_from_env=seed_extra_from_env, apply_yaml_bridge=apply_yaml_bridge)
    _shared._SECRETS = _SECRETS
    _mod("gateway.config", Platform=_Platform, PlatformConfig=_PlatformConfig)
    _mod("gateway.platforms.base", BasePlatformAdapter=_BasePlatformAdapter, SendResult=_SendResult,
         gateway_trust_env=lambda: False)
    _mod("gateway.platforms.base_streaming_tts", AudioFormat=_AudioFormat, StreamingTTSHandle=_StreamingTTSHandle)
    _mod("gateway.platforms.helpers", redact_phone=redact_phone)
    _mod("gateway.platforms.event", MessageEvent=_MessageEvent, MessageType=_MessageType)
    _mod("gateway.platforms.send_targets", StandaloneSendPolicy=_StandaloneSendPolicy)
    _mod("agent")
    _mod("agent.tts_provider", TTSProvider=_TTSProvider)
    _mod("agent.transcription_provider", TranscriptionProvider=_TranscriptionProvider,
         TranscriptionStreamSession=_TranscriptionStreamSession)


install_stubs()

_PKG_DIR = Path(__file__).resolve().parent.parent
_pkg = types.ModuleType("voxtral_voice_call")
_pkg.__path__ = [str(_PKG_DIR)]
sys.modules["voxtral_voice_call"] = _pkg

# pytest builds a Package collector for the plugin dir (it contains __init__.py) and imports
# that __init__.py as a bare "__init__" module, which breaks its relative import. Pre-seed the
# name so the collector gets an inert module instead; the real package is "voxtral_voice_call".
_bare_init = types.ModuleType("__init__")
_bare_init.__file__ = str(_PKG_DIR / "__init__.py")
_bare_init.__path__ = [str(_PKG_DIR)]
sys.modules.setdefault("__init__", _bare_init)
