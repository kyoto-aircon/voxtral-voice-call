"""Mistral Voxtral 3 clients: realtime STT websocket, realtime TTS websocket, batch speech over HTTPS.

Protocols follow the Mistral audio docs: realtime STT at
``wss://<base>/v1/audio/transcriptions/realtime?model=...`` with ``session.created`` /
``session.update`` / ``input_audio.append`` / ``transcription.text.delta`` / ``turn.*`` /
``transcription.done`` events; realtime TTS at ``wss://<base>/v1/audio/speech/realtime?model=...``
with ``session.update`` / ``input_text.append`` / ``input_text.flush`` and
``speech.audio.delta`` (base64 float32 PCM) until ``speech.audio.done``; batch TTS at
``POST <base>/v1/audio/speech``. Auth is always ``Authorization: Bearer MISTRAL_API_KEY``,
with the key read profile-scoped from the Hermes .env.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, Iterator, Optional

import threading

from gateway.platforms._shared import get_scoped_secret

from .audio import float32_to_pcm16


def mistral_api_key() -> str:
    return str(get_scoped_secret("MISTRAL_API_KEY", "") or "").strip()


def ws_base(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if base.startswith("https://"):
        return "wss://" + base[len("https://"):]
    if base.startswith("http://"):
        return "ws://" + base[len("http://"):]
    return "wss://" + base


def _headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


async def _ws_connect(url: str, headers: Dict[str, str], **kwargs: Any):
    try:
        from websockets.asyncio.client import connect as _connect
    except ImportError:
        from websockets import connect as _connect
    try:
        return await _connect(url, additional_headers=headers, **kwargs)
    except TypeError:
        return await _connect(url, extra_headers=headers, **kwargs)


def _ws_sync_connect(url: str, headers: Dict[str, str], **kwargs: Any):
    try:
        from websockets.sync.client import connect as _connect
    except ImportError:
        from websockets.legacy.sync.client import connect as _connect
    try:
        return _connect(url, additional_headers=headers, **kwargs)
    except TypeError:
        return _connect(url, extra_headers=headers, **kwargs)


class RealtimeSTT:
    """One realtime transcription session per phone call (async, lives on the gateway loop)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        sample_rate: int = 8000,
        delay_ms: int = 480,
        language: str = "",
        context_bias: tuple = (),
        turn_detection: Optional[dict] = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.delay_ms = delay_ms
        self.language = language
        self.context_bias = tuple(context_bias)
        self.turn_detection = turn_detection
        self.ws = None

    @property
    def url(self) -> str:
        return f"{ws_base(self.base_url)}/v1/audio/transcriptions/realtime?model={urllib.parse.quote(self.model)}"

    async def open(self) -> None:
        self.ws = await _ws_connect(self.url, _headers(self.api_key), max_size=None, ping_interval=20, ping_timeout=30)
        first = json.loads(await self.ws.recv())
        if first.get("type") == "error":
            raise RuntimeError(f"Mistral STT error: {first.get('error')}")
        if first.get("type") != "session.created":
            raise RuntimeError(f"Mistral STT: expected session.created, got {first.get('type')}")
        session: Dict[str, Any] = {
            "audio_format": {"encoding": "pcm_s16le", "sample_rate": self.sample_rate},
            "target_streaming_delay_ms": self.delay_ms,
        }
        if self.language:
            session["language"] = self.language
        if self.context_bias:
            session["context_bias"] = list(self.context_bias)
        session["turn_detection"] = self.turn_detection if self.turn_detection is not None else {}
        await self.ws.send(json.dumps({"type": "session.update", "session": session}))

    async def send_audio(self, pcm16: bytes) -> None:
        await self.ws.send(json.dumps({"type": "input_audio.append", "audio": base64.b64encode(pcm16).decode()}))

    async def end(self) -> None:
        await self.ws.send(json.dumps({"type": "input_audio.end"}))

    async def events(self) -> Any:
        async for raw in self.ws:
            yield json.loads(raw)

    async def close(self) -> None:
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None


class RealtimeTTSSession:
    """One PERSISTENT realtime TTS websocket, reused across sentences (gpt-live-style latency).

    ``stream_sentence`` sends ``input_text.append`` + ``input_text.flush`` on the already-open
    session and yields PCM until ``speech.audio.done`` — no reconnect per sentence, so
    time-to-first-audio stays at the model's ~90 ms instead of paying WS+session setup each time.
    """

    def __init__(self, *, base_url: str, model: str, api_key: str, voice: str, sample_rate: int = 24000) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.voice = voice
        self.sample_rate = sample_rate
        self._ws = None

    @property
    def url(self) -> str:
        return f"{ws_base(self.base_url)}/v1/audio/speech/realtime?model={urllib.parse.quote(self.model)}"

    def _connect(self) -> None:
        self._ws = _ws_sync_connect(self.url, _headers(self.api_key), max_size=None, ping_interval=15, ping_timeout=30)
        first = json.loads(self._ws.recv())
        if first.get("type") == "error":
            raise RuntimeError(f"Mistral TTS error: {first.get('error')}")
        self._ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "voice": self.voice,
                "audio_format": {"encoding": "pcm_f32le", "sample_rate": self.sample_rate},
            },
        }))

    def healthy(self) -> bool:
        return self._ws is not None

    def stream_sentence(self, text: str) -> Iterator[bytes]:
        """Yield int16 mono PCM at ``sample_rate`` for one sentence on the persistent session."""
        if self._ws is None:
            self._connect()
        assert self._ws is not None
        self._ws.send(json.dumps({"type": "input_text.append", "text": text}))
        self._ws.send(json.dumps({"type": "input_text.flush"}))
        for raw in self._ws:
            event = json.loads(raw)
            etype = event.get("type")
            if etype == "speech.audio.delta":
                payload = event.get("audio")
                if payload:
                    yield float32_to_pcm16(base64.b64decode(payload))
            elif etype == "speech.audio.done":
                break
            elif etype == "error":
                self.close()
                raise RuntimeError(f"Mistral TTS error: {event.get('error')}")

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None


class RealtimeTTSPool:
    """Pool of persistent realtime-TTS websockets keyed by (model, voice, rate); thread-safe.

    The speaker prefetches up to 3 sentences concurrently, so one shared socket would interleave
    frames — the pool hands each caller its own open session and reuses it afterwards.
    """

    def __init__(self, factory) -> None:
        self._factory = factory
        self._idle: list = []
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            while self._idle:
                session = self._idle.pop()
                if session.healthy():
                    return session
                session.close()
        return self._factory()

    def release(self, session) -> None:
        with self._lock:
            if session.healthy():
                self._idle.append(session)
            else:
                session.close()

    def close_all(self) -> None:
        with self._lock:
            while self._idle:
                self._idle.pop().close()


class RealtimeSTTSyncSession:
    """Blocking realtime STT session for worker threads (plugin TranscriptionStreamSession)."""

    def __init__(self, *, base_url: str, model: str, api_key: str, provider: str = "voxtral-realtime",
                 sample_rate: int = 16000, delay_ms: int = 480, language: str = "",
                 context_bias: tuple = ()) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.provider = provider
        self.sample_rate = sample_rate
        self.delay_ms = delay_ms
        self.language = language
        self.context_bias = tuple(context_bias)
        self._ws = None
        self._lock = threading.Lock()
        self._partial = ""
        self._final: Optional[str] = None
        self._error: Optional[str] = None
        self._done = threading.Event()

    @property
    def url(self) -> str:
        return f"{ws_base(self.base_url)}/v1/audio/transcriptions/realtime?model={urllib.parse.quote(self.model)}"

    def open(self) -> None:
        self._ws = _ws_sync_connect(self.url, _headers(self.api_key), max_size=None, ping_interval=20, ping_timeout=30)
        first = json.loads(self._ws.recv())
        if first.get("type") == "error":
            raise RuntimeError(f"Mistral STT error: {first.get('error')}")
        if first.get("type") != "session.created":
            raise RuntimeError(f"Mistral STT: expected session.created, got {first.get('type')}")
        session: Dict[str, Any] = {
            "audio_format": {"encoding": "pcm_s16le", "sample_rate": self.sample_rate},
            "target_streaming_delay_ms": self.delay_ms,
        }
        if self.language:
            session["language"] = self.language
        if self.context_bias:
            session["context_bias"] = list(self.context_bias)
        self._ws.send(json.dumps({"type": "session.update", "session": session}))
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self) -> None:
        try:
            for raw in self._ws:
                event = json.loads(raw)
                etype = event.get("type")
                if etype == "transcription.text.delta":
                    with self._lock:
                        self._partial += event.get("text") or ""
                elif etype == "transcription.done":
                    with self._lock:
                        self._final = event.get("text") or self._partial
                    self._done.set()
                    break
                elif etype == "error":
                    with self._lock:
                        self._error = str(event.get("error"))
                    self._done.set()
                    break
        except Exception as exc:
            with self._lock:
                if self._error is None:
                    self._error = str(exc)
            self._done.set()

    def push_audio(self, chunk: bytes) -> None:
        self._ws.send(json.dumps({"type": "input_audio.append", "audio": base64.b64encode(chunk).decode()}))

    def partial_transcript(self) -> str:
        with self._lock:
            return self._partial

    def finalize(self, timeout: float = 20.0) -> Dict[str, Any]:
        try:
            self._ws.send(json.dumps({"type": "input_audio.end"}))
        except Exception:
            pass
        self._done.wait(timeout)
        with self._lock:
            if self._error:
                return {"success": False, "transcript": "", "provider": self.provider, "error": self._error}
            return {"success": True, "transcript": (self._final or self._partial).strip(), "provider": self.provider}

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None


def batch_transcribe(
    *,
    base_url: str,
    model: str,
    api_key: str,
    file_path: str,
    provider: str = "voxtral-realtime",
    language: str = "",
    context_bias: tuple = (),
    diarize: bool = False,
    timeout: int = 180,
) -> Dict[str, Any]:
    """POST /v1/audio/transcriptions (multipart) and return the transcription envelope."""
    url = base_url.rstrip("/") + "/v1/audio/transcriptions"
    boundary = "----voxtral" + uuid.uuid4().hex
    with open(file_path, "rb") as fh:
        file_bytes = fh.read()
    filename = file_path.rsplit("/", 1)[-1] or "audio.wav"
    fields = [("model", model), ("diarize", "true" if diarize else "false")]
    if language:
        fields.append(("language", language))
    if context_bias:
        fields.append(("context_bias", ", ".join(context_bias)))
    parts = []
    for name, value in fields:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n".encode() + file_bytes + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        return {"success": True, "transcript": str(payload.get("text") or ""), "provider": provider}
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:500].decode("utf-8", "replace")
        return {"success": False, "transcript": "", "provider": provider,
                "error": f"Mistral transcribe HTTP {exc.code}: {detail}"}
    except Exception as exc:
        return {"success": False, "transcript": "", "provider": provider, "error": str(exc)}


def batch_speech(
    *,
    base_url: str,
    model: str,
    api_key: str,
    text: str,
    voice_id: str,
    output_format: str = "mp3",
    sample_rate: int = 24000,
    timeout: int = 120,
) -> bytes:
    """POST /v1/audio/speech (batch) and return the audio bytes."""
    url = base_url.rstrip("/") + "/v1/audio/speech"
    body = json.dumps({
        "model": model,
        "input": text,
        "voice_id": voice_id,
        "output_format": output_format,
        "sample_rate": sample_rate,
    }).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:500].decode("utf-8", "replace")
        raise RuntimeError(f"Mistral speech HTTP {exc.code}: {detail}") from exc
