"""Per-call state: Twilio media stream <-> Voxtral realtime STT <-> Hermes gateway turn."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any, Optional

from gateway.platforms.event import MessageEvent, MessageType

from . import audio as pcm_audio
from .settings import PHONE_SAMPLE_RATE
from .voxtral import RealtimeSTT, mistral_api_key

logger = logging.getLogger("voxtral-voice-call.call")

MEDIA_FRAME_MS = 20


class CallSession:
    """One phone call: owns the STT websocket, the transcript buffer and the outbound media writes."""

    def __init__(self, adapter: Any, *, call_sid: str, stream_sid: str, from_number: str, to_number: str, ws: Any) -> None:
        self.adapter = adapter
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.from_number = from_number
        self.to_number = to_number
        self.ws = ws
        self.ws_open = True
        self.stt: Optional[RealtimeSTT] = None
        self.transcript = ""
        self.turn_open = False
        self.tts_handle: Any = None
        self.last_mark: Optional[str] = None
        self.turn_count = 0
        self._turn_gen = 0
        self._last_delta = 0.0
        self._closed = False
        self._tasks: set = set()

    async def start(self) -> None:
        settings = self.adapter.settings
        api_key = mistral_api_key()
        if not api_key:
            logger.error("[voice-call] MISTRAL_API_KEY missing - cannot transcribe call %s", self.call_sid[:8])
            return
        self.stt = RealtimeSTT(
            base_url=settings.mistral_base_url,
            model=settings.stt_model,
            api_key=api_key,
            sample_rate=PHONE_SAMPLE_RATE,
            delay_ms=settings.stt_delay_ms,
            language=settings.stt_language,
            context_bias=settings.stt_context_bias,
            turn_detection=settings.turn_detection,
        )
        try:
            await self.stt.open()
        except Exception as exc:
            logger.error("[voice-call] STT session failed for call %s: %s", self.call_sid[:8], exc)
            self.stt = None
            return
        task = asyncio.create_task(self._stt_loop())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def on_audio(self, pcm16_8k: bytes) -> None:
        if self._closed or self.stt is None:
            return
        try:
            await self.stt.send_audio(pcm16_8k)
        except Exception as exc:
            logger.warning("[voice-call] STT send failed on call %s: %s", self.call_sid[:8], exc)
            self.stt = None

    async def _stt_loop(self) -> None:
        assert self.stt is not None
        try:
            async for event in self.stt.events():
                etype = event.get("type")
                if etype == "transcription.text.delta":
                    self.transcript += event.get("text") or ""
                    self._last_delta = time.monotonic()
                elif etype == "turn.start":
                    self.turn_open = True
                    self._turn_gen += 1
                    await self._on_barge_in()
                elif etype == "turn.end":
                    self.turn_open = False
                    gen = self._turn_gen
                    task = asyncio.create_task(self._finalize_turn(gen))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                elif etype == "transcription.done":
                    await self._finalize_turn(self._turn_gen, force=True)
                elif etype == "error":
                    logger.error("[voice-call] STT error on call %s: %s", self.call_sid[:8], event.get("error"))
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[voice-call] STT loop ended on call %s: %s", self.call_sid[:8], exc)

    async def _finalize_turn(self, gen: int, force: bool = False) -> None:
        grace = 0.4 if force else max(0.5, self.adapter.settings.stt_delay_ms / 1000.0)
        await asyncio.sleep(grace)
        if not force and (self.turn_open or self._turn_gen != gen or self._closed):
            return
        text = self.transcript.strip()
        self.transcript = ""
        if text:
            self._dispatch(text)

    def _dispatch(self, text: str) -> None:
        self.turn_count += 1
        message_id = f"{self.call_sid}:{self.turn_count}"
        source = self.adapter.build_source(
            chat_id=self.call_sid, chat_name=self.from_number, chat_type="dm",
            user_id=self.from_number, user_name=self.from_number, message_id=message_id,
        )
        event = MessageEvent(
            text=text, message_type=MessageType.VOICE, source=source,
            raw_message={"callSid": self.call_sid, "streamSid": self.stream_sid},
            message_id=message_id,
            metadata={"voice_call": True, "from_number": self.from_number},
        )
        logger.info("[voice-call] turn %d on call %s from %s: %s", self.turn_count, self.call_sid[:8], self.from_number[:6], text[:120])
        task = asyncio.create_task(self.adapter.handle_message(event))
        self.adapter._background_tasks.add(task)
        task.add_done_callback(self.adapter._background_tasks.discard)

    async def _on_barge_in(self) -> None:
        await self.send_clear()
        handle = self.tts_handle
        if handle is not None and not handle.aborted:
            await self.adapter.abort_streaming_tts(handle, error="caller interrupted")
        try:
            from tools.tts_streaming import mark_speech_interrupted
            mark_speech_interrupted()
        except Exception:
            pass

    async def send_pcm(self, pcm16: bytes, sample_rate: int) -> None:
        if not self.ws_open or self._closed or not pcm16:
            return
        pcm8 = pcm16 if sample_rate == PHONE_SAMPLE_RATE else pcm_audio.resample_pcm16(pcm16, sample_rate, PHONE_SAMPLE_RATE)
        mulaw = pcm_audio.mulaw_encode(pcm8)
        frame = PHONE_SAMPLE_RATE * MEDIA_FRAME_MS // 1000
        for piece in pcm_audio.chunk_bytes(mulaw, frame):
            if not self.ws_open or self._closed:
                return
            await self.ws.send_str(json.dumps({
                "event": "media", "streamSid": self.stream_sid,
                "media": {"payload": base64.b64encode(piece).decode()},
            }))

    async def speak_text(self, text: str) -> None:
        """Whole-reply fallback: stream the text through the plugin's realtime TTS into the call."""
        from .tts_provider import VoxtralRealtimeTTSProvider
        provider = VoxtralRealtimeTTSProvider()
        if not provider.is_available():
            logger.error("[voice-call] TTS provider unavailable (MISTRAL_API_KEY / websockets)")
            return
        iterator = iter(provider.stream(text))
        rate = provider.stream_sample_rate
        while True:
            chunk = await asyncio.to_thread(next, iterator, None)
            if chunk is None or self._closed:
                break
            await self.send_pcm(chunk, rate)

    async def send_clear(self) -> None:
        if self.ws_open and not self._closed:
            try:
                await self.ws.send_str(json.dumps({"event": "clear", "streamSid": self.stream_sid}))
            except Exception:
                pass

    async def send_mark(self, name: str) -> None:
        if self.ws_open and not self._closed:
            try:
                await self.ws.send_str(json.dumps({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": name}}))
            except Exception:
                pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        text = self.transcript.strip()
        if text:
            self._dispatch(text)
        if self.stt is not None:
            try:
                await self.stt.end()
            except Exception:
                pass
            await self.stt.close()
            self.stt = None
        for task in list(self._tasks):
            task.cancel()
        self.ws_open = False
