"""Unit tests for the voxtral-voice-call plugin (Hermes core stubbed in conftest)."""

from __future__ import annotations

import array
import asyncio
import base64
import hashlib
import hmac
import json

import pytest

from gateway.platforms import _shared as hermes_shared
from voxtral_voice_call import audio as pcm_audio
from voxtral_voice_call import settings as vc_settings
from voxtral_voice_call.adapter import (
    VoiceCallAdapter,
    _port_variant,
    _twilio_signature_ok,
    _twiml_connect,
    _twiml_hangup,
)
from voxtral_voice_call.call_session import CallSession
from voxtral_voice_call.stt_provider import VoxtralRealtimeSTTProvider
from voxtral_voice_call.tts_provider import VoxtralRealtimeTTSProvider
from voxtral_voice_call import voxtral as voxtral_mod


# ---------------------------------------------------------------- audio


def test_mulaw_roundtrip():
    samples = array.array("h", [0, 100, -100, 3000, -3000, 32767, -32768, 12345])
    pcm = samples.tobytes()
    decoded = array.array("h")
    decoded.frombytes(pcm_audio.mulaw_decode(pcm_audio.mulaw_encode(pcm)))
    assert len(decoded) == len(samples)
    for orig, dec in zip(samples, decoded):
        assert abs(orig - dec) <= max(1024, abs(orig) // 8)


def test_resample_length_and_identity():
    pcm = array.array("h", list(range(0, 4800, 10))).tobytes()
    assert pcm_audio.resample_pcm16(pcm, 24000, 24000) == pcm
    out = pcm_audio.resample_pcm16(pcm, 24000, 8000)
    assert len(out) == 2 * (len(pcm) // 2) // 3
    assert pcm_audio.resample_pcm16(b"", 24000, 8000) == b""


def test_float32_to_pcm16():
    import struct
    data = struct.pack("<4f", 0.0, 1.0, -1.0, 0.5)
    out = array.array("h")
    out.frombytes(pcm_audio.float32_to_pcm16(data))
    assert list(out) == [0, 32767, -32767, 16383]


# ---------------------------------------------------------------- settings


def _settings(extra=None):
    return vc_settings.VoiceCallSettings.from_extra(extra or {})


def test_settings_defaults(monkeypatch):
    for var in list(vc_settings.__dict__):
        pass
    monkeypatch.delenv("VOICE_CALL_WEBHOOK_PORT", raising=False)
    s = _settings({})
    assert s.webhook_port == vc_settings.DEFAULT_WEBHOOK_PORT
    assert s.stt_model == "voxtral-mini-transcribe-realtime-3-0"
    assert s.tts_realtime_model == "voxtral-mini-tts-3-0"
    assert s.mistral_base_url == "https://api.mistral.ai"
    assert s.caller_allowed("+15551234567") is True


def test_settings_env_and_yaml(monkeypatch):
    monkeypatch.setenv("VOICE_CALL_WEBHOOK_PORT", "9999")
    monkeypatch.setenv("VOICE_CALL_ALLOWED_NUMBERS", "+1111,+2222")
    s = _settings({"stt_delay_ms": 240})
    assert s.webhook_port == 9999
    assert s.stt_delay_ms == 240
    assert s.allowed_numbers == frozenset({"+1111", "+2222"})
    assert s.caller_allowed("+1111") is True
    assert s.caller_allowed("+3333") is False
    monkeypatch.setenv("VOICE_CALL_ALLOW_ALL_USERS", "true")
    s2 = _settings({})
    assert s2.caller_allowed("+3333") is True


def test_settings_fall_back_to_plugin_settings(monkeypatch):
    import voxtral_voice_call.settings as st
    monkeypatch.delenv("VOICE_CALL_STT_DELAY_MS", raising=False)
    monkeypatch.delenv("VOICE_CALL_TTS_VOICE", raising=False)
    monkeypatch.setattr(st, "_plugin_settings", lambda: {"stt_delay_ms": 240, "tts_voice": "gb_jane_neutral"})
    s = _settings({})
    assert s.stt_delay_ms == 240
    assert s.tts_voice == "gb_jane_neutral"
    monkeypatch.setenv("VOICE_CALL_STT_DELAY_MS", "900")
    assert _settings({}).stt_delay_ms == 900


# ---------------------------------------------------------------- twilio helpers


def test_twilio_signature():
    token = "test-token"
    url = "https://example.com/webhooks/twilio/voice"
    params = {"CallSid": "CA123", "From": "+1555"}
    payload = url + "".join(f"{k}{v}" for k, v in sorted(params.items()))
    sig = base64.b64encode(hmac.new(token.encode(), payload.encode(), hashlib.sha1).digest()).decode()
    assert _twilio_signature_ok(token, url, params, sig) is True
    assert _twilio_signature_ok(token, url, params, "bogus") is False
    assert _twilio_signature_ok("wrong", url, params, sig) is False


def test_port_variant():
    assert _port_variant("https://example.com:443/webhooks/twilio/voice") == "https://example.com/webhooks/twilio/voice"
    assert _port_variant("https://example.com:8443/x") == "https://example.com/x"
    assert _port_variant("https://example.com/x") is None


def test_twiml_connect():
    xml = _twiml_connect("wss://example.com/stream", "This call may be recorded.")
    assert '<Stream url="wss://example.com/stream"/>' in xml
    assert "<Say>This call may be recorded.</Say>" in xml
    assert xml.startswith("<?xml")
    xml2 = _twiml_connect("wss://example.com/stream")
    assert "<Say>" not in xml2
    assert "<Hangup/>" in _twiml_hangup("nope")


# ---------------------------------------------------------------- adapter construction


def _adapter(extra=None):
    from gateway.config import PlatformConfig
    return VoiceCallAdapter(PlatformConfig(enabled=True, extra=extra or {}))


def test_adapter_registers_platform_name():
    adapter = _adapter({})
    assert str(adapter.platform) == "voice-call"
    assert adapter.serves_profile_prefix is True


def test_stream_ws_url():
    adapter = _adapter({"webhook_url": "https://example.com/webhooks/twilio/voice"})
    request = type("R", (), {"scheme": "https", "host": "ignored"})()
    assert adapter._stream_ws_url(request) == "wss://example.com/stream"
    adapter2 = _adapter({"webhook_url": "http://localhost:8081/webhooks/twilio/voice"})
    assert adapter2._stream_ws_url(request) == "ws://localhost:8081/stream"


def test_signature_validation_uses_token(monkeypatch):
    hermes_shared._SECRETS["TWILIO_AUTH_TOKEN"] = "tok"
    adapter = _adapter({"webhook_url": "https://example.com/webhooks/twilio/voice"})
    form = {"CallSid": ["CA1"], "From": ["+1555"]}
    flat = {k: v[0] for k, v in form.items()}
    payload = adapter._settings.webhook_url + "".join(f"{k}{v}" for k, v in sorted(flat.items()))
    sig = base64.b64encode(hmac.new(b"tok", payload.encode(), hashlib.sha1).digest()).decode()
    request = type("R", (), {"headers": {"X-Twilio-Signature": sig}})()
    assert adapter._validate_signature(request, form) is True
    bad = type("R", (), {"headers": {"X-Twilio-Signature": "nope"}})()
    assert adapter._validate_signature(bad, form) is False
    hermes_shared._SECRETS.clear()


# ---------------------------------------------------------------- call session turn flow


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_str(self, data):
        self.sent.append(json.loads(data))


class FakeSTT:
    def __init__(self, events):
        self._events = events
        self.sent_audio = []
        self.ended = False

    async def send_audio(self, pcm):
        self.sent_audio.append(pcm)

    async def events(self):
        for ev in self._events:
            yield ev

    async def end(self):
        self.ended = True

    async def close(self):
        pass


def _make_session(adapter, events):
    ws = FakeWS()
    call = CallSession(adapter, call_sid="CA999", stream_sid="MZ1", from_number="+1555", to_number="+1666", ws=ws)
    call.stt = FakeSTT(events)
    return call, ws


def test_turn_end_dispatches_voice_event():
    adapter = _adapter({"stt_delay_ms": 80})
    events = [
        {"type": "transcription.text.delta", "text": "hello "},
        {"type": "transcription.text.delta", "text": "there"},
        {"type": "turn.end", "time_s": 1.2, "end_reason": "threshold"},
        {"type": "transcription.done", "text": "hello there"},
    ]
    call, ws = _make_session(adapter, events)

    async def run():
        task = asyncio.create_task(call._stt_loop())
        await asyncio.sleep(1.2)
        await call.close()
        await task

    asyncio.run(run())
    assert len(adapter.dispatched) == 1
    event = adapter.dispatched[0]
    assert event.text == "hello there"
    assert str(event.message_type) == "MessageType.VOICE" or event.message_type.value == "voice"
    assert event.source.chat_id == "CA999"
    assert event.metadata["voice_call"] is True


def test_barge_in_clears_media_and_aborts_tts():
    adapter = _adapter({"stt_delay_ms": 80})
    events = [
        {"type": "turn.start", "time_s": 0.0},
        {"type": "turn.end", "time_s": 1.0, "end_reason": "threshold"},
        {"type": "transcription.text.delta", "text": "hi"},
        {"type": "turn.start", "time_s": 2.0},
        {"type": "transcription.done", "text": "hi"},
    ]
    call, ws = _make_session(adapter, events)

    class FakeHandle:
        aborted = False
        chat_id = "CA999"

    handle = FakeHandle()
    call.tts_handle = handle

    async def run():
        task = asyncio.create_task(call._stt_loop())
        await asyncio.sleep(1.2)
        await call.close()
        await task

    asyncio.run(run())
    clears = [m for m in ws.sent if m.get("event") == "clear"]
    assert clears, "expected a clear frame on barge-in"
    assert handle.aborted is True
    assert adapter.dispatched and adapter.dispatched[0].text == "hi"


def test_send_pcm_emits_media_frames_8khz_mulaw():
    adapter = _adapter({})
    call, ws = _make_session(adapter, [])
    pcm = array.array("h", [0] * 2400).tobytes()  # 100 ms at 24 kHz
    asyncio.run(call.send_pcm(pcm, 24000))
    media = [m for m in ws.sent if m.get("event") == "media"]
    assert media, "expected media frames"
    total_mulaw = sum(len(base64.b64decode(m["media"]["payload"])) for m in media)
    assert total_mulaw == 800  # 100 ms at 8 kHz = 800 mu-law samples
    assert all(m["streamSid"] == "MZ1" for m in media)


# ---------------------------------------------------------------- tts provider


def test_provider_identity_and_streaming():
    provider = VoxtralRealtimeTTSProvider()
    assert provider.name == "voxtral-realtime"
    assert provider.streams_pcm is True
    assert provider.stream_sample_rate == vc_settings.DEFAULT_TTS_SAMPLE_RATE
    hermes_shared._SECRETS.clear()
    assert provider.is_available() is False
    hermes_shared._SECRETS["MISTRAL_API_KEY"] = "k"
    import importlib.util
    if importlib.util.find_spec("websockets") is not None:
        assert provider.is_available() is True
    hermes_shared._SECRETS.clear()


def test_provider_stream_empty_text():
    provider = VoxtralRealtimeTTSProvider()
    assert list(provider.stream("   ")) == []


def test_resolve_voice_id_accepts_ids_and_labels():
    from voxtral_voice_call.tts_provider import resolve_voice_id, VOXTRAL3_VOICES
    assert resolve_voice_id("voice_1788") == "voice_1788"
    assert resolve_voice_id("David (en-GB)") == "voice_1788"
    assert resolve_voice_id("en_paul_neutral") == "en_paul_neutral"
    assert resolve_voice_id("") == "en_paul_neutral"
    assert resolve_voice_id("custom-clone-xyz") == "custom-clone-xyz"
    assert len(VOXTRAL3_VOICES) >= 35


def test_tts_pool_reuses_persistent_connection(monkeypatch):
    import queue as queue_mod
    import struct
    from voxtral_voice_call import tts_provider as tp

    connects = []

    class FakeTTSWS:
        def __init__(self):
            self._q = queue_mod.Queue()
            self._q.put(json.dumps({"type": "session.created", "session": {}}))
            self._q.put(json.dumps({"type": "session.updated", "session": {}}))

        def recv(self):
            return self._q.get(timeout=5)

        def send(self, data):
            msg = json.loads(data)
            if msg.get("type") == "input_text.flush":
                pcm = struct.pack("<2f", 0.5, -0.5)
                self._q.put(json.dumps({"type": "speech.audio.delta",
                                         "audio_data": base64.b64encode(pcm).decode()}))
                self._q.put(json.dumps({"type": "speech.audio.done"}))

        def __iter__(self):
            while True:
                item = self._q.get(timeout=5)
                yield item
                if json.loads(item).get("type") == "speech.audio.done":
                    return

        def close(self):
            pass

    def fake_connect(*a, **kw):
        ws = FakeTTSWS()
        connects.append(ws)
        return ws

    monkeypatch.setattr(voxtral_mod, "_ws_sync_connect", fake_connect)
    hermes_shared._SECRETS["MISTRAL_API_KEY"] = "k"
    tp._POOLS.clear()
    provider = tp.VoxtralRealtimeTTSProvider()
    out1 = b"".join(provider.stream("First sentence."))
    out2 = b"".join(provider.stream("Second sentence."))
    assert len(out1) == 4 and len(out2) == 4
    assert len(connects) == 1, "expected one persistent websocket reused across sentences"
    hermes_shared._SECRETS.clear()
    tp._POOLS.clear()


# ---------------------------------------------------------------- stt provider


def test_stt_provider_identity():
    provider = VoxtralRealtimeSTTProvider()
    assert provider.name == "voxtral-realtime"
    assert provider.streaming_capable is True
    hermes_shared._SECRETS.clear()
    assert provider.is_available() is False
    hermes_shared._SECRETS["MISTRAL_API_KEY"] = "k"
    import importlib.util
    if importlib.util.find_spec("websockets") is not None:
        assert provider.is_available() is True
    hermes_shared._SECRETS.clear()


def test_stt_transcribe_envelope(monkeypatch, tmp_path):
    import voxtral_voice_call.stt_provider as sp
    provider = VoxtralRealtimeSTTProvider()
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFfake")
    monkeypatch.setattr(sp, "batch_transcribe",
                        lambda **kw: {"success": True, "transcript": "hi there", "provider": kw["provider"]})
    result = provider.transcribe(str(wav), language="en")
    assert result == {"success": True, "transcript": "hi there", "provider": "voxtral-realtime"}


class FakeSyncWS:
    def __init__(self, events):
        import queue
        self._queue = queue.Queue()
        for ev in events:
            self._queue.put(json.dumps(ev))
        self.sent = []
        self.closed = False

    def recv(self):
        return self._queue.get(timeout=5)

    def send(self, data):
        msg = json.loads(data)
        self.sent.append(msg)
        if msg.get("type") == "input_audio.end":
            self._queue.put(json.dumps({"type": "transcription.done", "text": "final text"}))

    def __iter__(self):
        import queue
        while True:
            try:
                yield self._queue.get(timeout=5)
            except queue.Empty:
                if self.closed:
                    return

    def close(self):
        self.closed = True


def test_stt_stream_session(monkeypatch):
    fake = FakeSyncWS([
        {"type": "session.created", "session": {}},
        {"type": "transcription.text.delta", "text": "hel"},
        {"type": "transcription.text.delta", "text": "lo"},
    ])
    monkeypatch.setattr(voxtral_mod, "_ws_sync_connect", lambda *a, **kw: fake)
    hermes_shared._SECRETS["MISTRAL_API_KEY"] = "k"
    provider = VoxtralRealtimeSTTProvider()
    session = provider.open_stream_session(language="en")
    session.push_audio(b"\x00" * 320)
    import time
    deadline = time.time() + 5
    while time.time() < deadline and session.partial_transcript() != "hello":
        time.sleep(0.02)
    assert session.partial_transcript() == "hello"
    appends = [m for m in fake.sent if m["type"] == "input_audio.append"]
    assert len(appends) == 1
    result = session.finalize()
    assert result == {"success": True, "transcript": "final text", "provider": "voxtral-realtime"}
    hermes_shared._SECRETS.clear()


# ---------------------------------------------------------------- plugin registration


def test_register_wires_platform_and_tts_provider():
    from voxtral_voice_call import plugin as plugin_mod

    calls = {}

    class Ctx:
        def register_platform(self, **kwargs):
            calls["platform"] = kwargs

        def register_tts_provider(self, provider):
            calls["tts_provider"] = provider

        def register_transcription_provider(self, provider):
            calls["stt_provider"] = provider

    plugin_mod.register(Ctx())
    assert calls["platform"]["name"] == "voice-call"
    assert calls["platform"]["adapter_factory"] is VoiceCallAdapter
    assert "MISTRAL_API_KEY" in calls["platform"]["required_env"]
    assert calls["platform"]["env_enable_credentials"] == ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN")
    assert calls["platform"]["allowed_users_env"] == "VOICE_CALL_ALLOWED_NUMBERS"
    assert calls["tts_provider"].name == "voxtral-realtime"
    assert calls["stt_provider"].name == "voxtral-realtime"
