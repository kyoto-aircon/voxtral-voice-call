"""Voice-call platform adapter: Twilio inbound calls <-> Hermes agent, with Voxtral 3 STT/TTS.

Inbound: Twilio POSTs the call to ``/webhooks/twilio/voice`` (signature-validated); the TwiML
answer opens a Media Streams websocket at ``/stream``. Caller audio (8 kHz mu-law) is forwarded
to Voxtral Mini Transcribe Realtime 3; ``turn.end`` dispatches the transcript to the gateway as a
VOICE message on a per-call session. Outbound: the gateway's streaming-TTS consumer calls
``begin/write/finish_streaming_tts`` with PCM chunks while the LLM generates; we resample to
8 kHz mu-law and push ``media`` frames into the call. ``turn.start`` triggers barge-in
(``clear`` + abort the running TTS stream).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import json
import logging
import time
import urllib.parse
from typing import Any, Optional
from xml.sax.saxutils import escape as xml_escape, quoteattr as xml_quoteattr

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.base_streaming_tts import AudioFormat, StreamingTTSHandle
from gateway.platforms.helpers import redact_phone

from . import audio as pcm_audio
from .call_session import CallSession
from .settings import VoiceCallSettings

logger = logging.getLogger("voxtral-voice-call")

MAX_TWILIO_BODY_BYTES = 65_536
VOICE_WEBHOOK_PATH = "/webhooks/twilio/voice"
STREAM_PATH = "/stream"


def check_requirements() -> bool:
    return importlib.util.find_spec("aiohttp") is not None and importlib.util.find_spec("websockets") is not None


def is_connected(pconfig: Any) -> bool:
    from gateway.platforms._shared import get_scoped_secret
    return bool(
        get_scoped_secret("TWILIO_ACCOUNT_SID", "")
        and get_scoped_secret("TWILIO_AUTH_TOKEN", "")
        and get_scoped_secret("MISTRAL_API_KEY", "")
    )


def _twilio_signature_ok(auth_token: str, url: str, params: dict, signature: str) -> bool:
    payload = url + "".join(f"{k}{v}" for k, v in sorted(params.items()))
    expected = base64.b64encode(hmac.new(auth_token.encode(), payload.encode(), hashlib.sha1).digest()).decode()
    return hmac.compare_digest(expected, signature or "")


def _port_variant(url: str) -> Optional[str]:
    parts = urllib.parse.urlsplit(url)
    if not parts.port:
        return None
    netloc = parts.hostname or ""
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _twiml(body: str) -> str:
    return '<?xml version="1.0" encoding="UTF-8"?><Response>' + body + "</Response>"


def _twiml_connect(stream_url: str, consent_text: str = "") -> str:
    say = f"<Say>{xml_escape(consent_text)}</Say>" if consent_text else ""
    return _twiml(say + f"<Connect><Stream url={xml_quoteattr(stream_url)}/></Connect>")


def _twiml_hangup(message: str) -> str:
    return _twiml(f"<Say>{xml_escape(message)}</Say><Hangup/>")


class VoiceCallAdapter(BasePlatformAdapter):
    """Twilio voice calls <-> Hermes: one gateway session per call SID; replies spoken via streaming TTS."""

    serves_profile_prefix: bool = True

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("voice-call"))
        self._settings = VoiceCallSettings.from_extra(config.extra or {})
        self._calls: dict[str, CallSession] = {}
        self._runner = None

    @property
    def settings(self) -> VoiceCallSettings:
        return self._settings

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        s = self._settings
        fatal = None
        if not s.from_number:
            fatal = ("voice_call_missing_phone_number", "[voice-call] TWILIO_PHONE_NUMBER not set - cannot receive calls")
        elif not s.mistral_api_key():
            fatal = ("voice_call_missing_mistral_key",
                     "[voice-call] MISTRAL_API_KEY not set in ~/.hermes/.env - get one at https://console.mistral.ai/")
        elif not s.webhook_url and not s.insecure_no_signature:
            fatal = ("voice_call_missing_webhook_url",
                     "[voice-call] VOICE_CALL_WEBHOOK_URL is required for Twilio signature validation "
                     "(set it to the public URL of /webhooks/twilio/voice configured in the Twilio console). "
                     "For local dev only, set VOICE_CALL_INSECURE_NO_SIGNATURE=true.")
        if fatal:
            logger.error(fatal[1])
            self._set_fatal_error(fatal[0], fatal[1], retryable=False)
            return False
        if s.insecure_no_signature and not s.webhook_url:
            logger.warning("[voice-call] VOICE_CALL_INSECURE_NO_SIGNATURE=true - signature validation DISABLED, dev only")
        from aiohttp import web
        from gateway.platforms.shared_ingress import bind_listener
        app = web.Application(client_max_size=MAX_TWILIO_BODY_BYTES)
        app.router.add_post(VOICE_WEBHOOK_PATH, self._handle_voice_webhook)
        app.router.add_get(STREAM_PATH, self._handle_media_stream)
        app.router.add_get("/health", lambda _: web.Response(text="ok"))
        self._runner = await bind_listener(self, app, s.webhook_host, s.webhook_port, VOICE_WEBHOOK_PATH)
        self._running = True
        if self._runner is not None:
            logger.info("[voice-call] webhook + media-stream server on %s:%d, from: %s",
                        s.webhook_host, s.webhook_port, redact_phone(s.from_number))
        self._wire_plugin_handlers(None)
        return True

    async def disconnect(self) -> None:
        for call in list(self._calls.values()):
            await call.close()
        self._calls.clear()
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._running = False
        logger.info("[voice-call] disconnected")

    def _stream_ws_url(self, request: Any) -> str:
        s = self._settings
        base = s.webhook_url
        if not base:
            base = f"{request.scheme}://{request.host}"
        parts = urllib.parse.urlsplit(base)
        scheme = "wss" if parts.scheme == "https" else "ws"
        netloc = parts.netloc
        return f"{scheme}://{netloc}{STREAM_PATH}"

    def _validate_signature(self, request: Any, form: dict) -> bool:
        s = self._settings
        if s.insecure_no_signature:
            return True
        signature = request.headers.get("X-Twilio-Signature", "")
        if not signature or not s.webhook_url:
            return False
        flat = {k: v[0] for k, v in form.items() if v}
        token = self._twilio_auth_token()
        if _twilio_signature_ok(token, s.webhook_url, flat, signature):
            return True
        variant = _port_variant(s.webhook_url)
        return bool(variant) and _twilio_signature_ok(token, variant, flat, signature)

    def _twilio_auth_token(self) -> str:
        from gateway.platforms._shared import get_scoped_secret
        return str(get_scoped_secret("TWILIO_AUTH_TOKEN", "") or "").strip()

    async def _handle_voice_webhook(self, request: Any) -> Any:
        from aiohttp import web
        try:
            raw = await request.read()
            if len(raw) > MAX_TWILIO_BODY_BYTES:
                return web.Response(status=413, text="payload too large")
            form = urllib.parse.parse_qs(raw.decode("utf-8"), keep_blank_values=True)
        except Exception as exc:
            logger.error("[voice-call] webhook parse error: %s", exc)
            return web.Response(status=400, text="bad request")
        if not self._validate_signature(request, form):
            logger.warning("[voice-call] rejected webhook: invalid X-Twilio-Signature")
            return web.Response(status=403, text="invalid signature")
        call_sid = (form.get("CallSid") or [""])[0].strip()
        from_number = (form.get("From") or [""])[0].strip()
        to_number = (form.get("To") or [""])[0].strip()
        if not call_sid or not from_number:
            return web.Response(text=_twiml(""), content_type="application/xml")
        if not self._settings.caller_allowed(from_number):
            logger.warning("[voice-call] rejected call from unauthorized number %s", redact_phone(from_number))
            return web.Response(text=_twiml_hangup("This number is not authorized to use this agent."), content_type="application/xml")
        logger.info("[voice-call] inbound call %s from %s to %s", call_sid[:8], redact_phone(from_number), redact_phone(to_number))
        stream_url = self._stream_ws_url(request)
        return web.Response(text=_twiml_connect(stream_url, self._settings.consent_text), content_type="application/xml")

    async def _handle_media_stream(self, request: Any) -> Any:
        from aiohttp import web
        ws = web.WebSocketResponse(heartbeat=25, max_msg_size=MAX_TWILIO_BODY_BYTES)
        await ws.prepare(request)
        call: Optional[CallSession] = None
        started = time.monotonic()
        async for msg in ws:
            if msg.type != web.WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except Exception:
                continue
            event = data.get("event")
            if event == "start":
                start = data.get("start") or {}
                call_sid = str(start.get("callSid") or "")
                stream_sid = str(start.get("streamSid") or "")
                from_number = str(start.get("from") or "")
                if not self._settings.caller_allowed(from_number):
                    logger.warning("[voice-call] media stream from unauthorized number %s - closing", redact_phone(from_number))
                    break
                call = CallSession(self, call_sid=call_sid, stream_sid=stream_sid,
                                   from_number=from_number, to_number=str(start.get("to") or ""), ws=ws)
                self._calls[call_sid] = call
                await call.start()
            elif event == "media" and call is not None:
                payload = (data.get("media") or {}).get("payload") or ""
                if payload:
                    try:
                        pcm = pcm_audio.mulaw_decode(base64.b64decode(payload))
                    except Exception:
                        continue
                    await call.on_audio(pcm)
            elif event == "mark" and call is not None:
                call.last_mark = (data.get("mark") or {}).get("name")
            elif event == "stop":
                if call is not None:
                    await call.close()
                    self._calls.pop(call.call_sid, None)
                    call = None
            if self._settings.max_call_seconds and time.monotonic() - started > self._settings.max_call_seconds:
                logger.info("[voice-call] call %s exceeded max_call_seconds=%d - closing", (call.call_sid[:8] if call else "?"), self._settings.max_call_seconds)
                break
        if call is not None:
            await call.close()
            self._calls.pop(call.call_sid, None)
        return ws

    def supports_streaming_tts(self, chat_id: str, audio_format: AudioFormat) -> bool:
        call = self._calls.get(str(chat_id))
        return call is not None and call.ws_open and audio_format.channels == 1 and audio_format.sample_width == 2

    async def begin_streaming_tts(self, chat_id: str, audio_format: AudioFormat, metadata: Optional[dict] = None) -> Optional[StreamingTTSHandle]:
        call = self._calls.get(str(chat_id))
        if call is None or not call.ws_open:
            return None
        handle = StreamingTTSHandle(chat_id=str(chat_id), audio_format=audio_format)
        call.tts_handle = handle
        return handle

    async def write_streaming_tts(self, handle: StreamingTTSHandle, chunk: bytes) -> None:
        call = self._calls.get(handle.chat_id)
        if call is None or handle.aborted or not chunk:
            return
        await call.send_pcm(chunk, handle.audio_format.sample_rate)

    async def finish_streaming_tts(self, handle: StreamingTTSHandle, *, interrupted: bool = False) -> None:
        call = self._calls.get(handle.chat_id)
        if call is None:
            return
        if call.tts_handle is handle:
            call.tts_handle = None
        await call.send_mark("tts_done")

    async def abort_streaming_tts(self, handle: StreamingTTSHandle, error: Optional[str] = None) -> None:
        handle.aborted = True
        call = self._calls.get(handle.chat_id)
        if call is None:
            return
        if call.tts_handle is handle:
            call.tts_handle = None
        await call.send_clear()

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None, metadata: Optional[dict] = None, **kwargs: Any) -> SendResult:
        call = self._calls.get(str(chat_id))
        if call is None:
            return SendResult(success=False, error="voice-call: no active call for this chat")
        await call.speak_text(content)
        return SendResult(success=True)

    async def send_voice(self, chat_id: str, audio_path: str, **kwargs: Any) -> SendResult:
        call = self._calls.get(str(chat_id))
        if call is None:
            return SendResult(success=False, error="voice-call: no active call for this chat")
        pcm, rate = pcm_audio.read_wav(audio_path)
        if pcm is None:
            return SendResult(success=False, error="voice-call: whole-file auto-TTS needs a PCM WAV file")
        await call.send_pcm(pcm, rate)
        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"chat_id": str(chat_id), "title": "Voice call", "type": "dm"}

    def format_message(self, content: str) -> str:
        return content
