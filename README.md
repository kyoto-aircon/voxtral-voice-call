# voxtral-voice-call — call Hermes and talk to it fluidly

A standalone Hermes plugin that gives your agent a phone number. Callers dial your Twilio
number and have a real-time voice conversation with Hermes: caller speech is streamed to
**Voxtral Mini Transcribe Realtime 3** (server-side turn detection), each finished turn runs
through the normal Hermes agent (tools, memory, skills, per-call session), and the reply is
spoken back with **Voxtral Mini TTS Realtime 3** through the gateway's streaming-TTS pipeline
(sentence-level, ~90 ms time-to-first-audio) with barge-in support.

No Hermes core changes — everything ships as this plugin: a `voice-call` gateway platform
adapter plus a `voxtral-realtime` TTS provider.

## Architecture

```
caller ──PSTN──► Twilio ──POST /webhooks/twilio/voice──► adapter (TwiML: <Connect><Stream>)
                  ▲                                        │ wss /stream (Media Streams, 8 kHz mu-law)
                  │ media frames ◄── PCM ◄── Voxtral Mini TTS Realtime 3 ◄── LLM deltas
                  │                                        ▲         (gateway StreamingTTSConsumer)
                  │                                        │ turn transcript (MessageType.VOICE)
                  └── Hermes agent session per CallSid ◄───┴── Voxtral Mini Transcribe Realtime 3
```

- **Inbound**: `turn.end` from the realtime STT dispatches the transcript via
  `handle_message(MessageEvent(message_type=VOICE, ...))` with `chat_id = CallSid`, so one call
  is one continuous agent session. `turn.start` triggers barge-in: Twilio `clear` + abort of
  the in-flight TTS stream.
- **Outbound**: the adapter implements the streaming-TTS contract (`supports_streaming_tts` /
  `begin_streaming_tts` / `write_streaming_tts` / `finish_streaming_tts` / `abort_streaming_tts`),
  so the gateway feeds LLM text deltas straight into the Voxtral realtime TTS websocket —
  playback starts mid-generation. PCM is resampled to 8 kHz and encoded to mu-law for the call.
- **TTS provider**: `voxtral-realtime` (plugin `TTSProvider` with `streams_pcm`) also upgrades
  CLI/TUI/gateway voice replies when selected.

## Install

```bash
hermes plugins install <owner>/voxtral-voice-call
```

Dependencies (`aiohttp`, `websockets`) are declared in `plugin.yaml`.

## Configuration

Secrets live in `~/.hermes/.env` (the `MISTRAL_API_KEY` template is already there):

```dotenv
MISTRAL_API_KEY=...            # Voxtral 3 realtime STT/TTS (console.mistral.ai)
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...          # also validates X-Twilio-Signature
TWILIO_PHONE_NUMBER=+1555...   # voice-capable number callers dial
VOICE_CALL_WEBHOOK_URL=https://your-host/webhooks/twilio/voice   # public URL, set in Twilio console
VOICE_CALL_ALLOWED_NUMBERS=+1555...   # recommended allowlist (E.164, comma-separated)
```

Optional: `VOICE_CALL_WEBHOOK_HOST`/`_PORT` (default `127.0.0.1:8081`), `VOICE_CALL_CONSENT_TEXT`
(spoken `<Say>` before connecting), `VOICE_CALL_MAX_CALL_SECONDS` (cost cap, default 1800),
`VOICE_CALL_MISTRAL_BASE_URL` (use an EU endpoint for zero-retention), `VOICE_CALL_STT_MODEL`,
`VOICE_CALL_STT_DELAY_MS` (80–1200 or 2400; default 480), `VOICE_CALL_STT_LANGUAGE`,
`VOICE_CALL_STT_CONTEXT_BIAS`, `VOICE_CALL_TTS_MODEL`, `VOICE_CALL_TTS_VOICE`,
`VOICE_CALL_TTS_VOICE_ID`, `VOICE_CALL_TTS_SAMPLE_RATE`, `VOICE_CALL_INSECURE_NO_SIGNATURE`
(dev only).

`config.yaml`:

```yaml
platforms:
  voice-call:
    enabled: true
voice:
  auto_tts: true            # required: lets the gateway stream TTS for voice-input turns
tts:
  provider: voxtral-realtime # use the Voxtral 3 voice everywhere (streaming)
# tts.streaming.provider: voxtral-realtime  # pin the streamer explicitly if needed
```

Any `platforms.voice-call.*` key above is also settable directly in that YAML block (bridged to
the env, env wins).

Twilio console: point the number's **Voice URL** at `https://your-host/webhooks/twilio/voice`
(POST). Put the host behind TLS (the media stream is `wss://your-host/stream`).

## Safety

- `X-Twilio-Signature` validated on every webhook (HMAC-SHA1 over URL + sorted params, incl.
  default-port variants), unless the explicit dev-only insecure flag is set.
- Caller allowlist enforced both at the webhook (TwiML hangup) and by gateway authz
  (`allowed_users_env`), with `VOICE_CALL_ALLOW_ALL_USERS` as the explicit open switch.
- Optional consent announcement, per-call duration cap, phone numbers redacted in logs
  (`pii_safe`), secrets read via the profile-scoped secret scope — never `os.environ`, never
  logged.

## Costs (Mistral API, public preview pricing)

Realtime transcription ≈ $0.008/audio-minute, realtime TTS ≈ $13 per 1M characters, Twilio
per-minute telephony + media. Set `VOICE_CALL_MAX_CALL_SECONDS` and the allowlist to bound spend.

## Limitations / next steps

- Transport is Twilio Media Streams; a LiveKit transport (Mistral's voice-agent SDK style) can
  be added behind a `transport:` config key later.
- Whole-file auto-TTS fallback (`send_voice`) accepts PCM WAV only.
- Realtime TTS websocket event field names (`speech.audio.delta.audio` as base64 float32) follow
  the public preview protocol; if Mistral renames fields in GA, only `voxtral.py` changes.

## Development

```bash
python -m py_compile *.py
pytest tests/
hermes plugins validate /path/to/voxtral-voice-call   # from a hermes-agent checkout
hermes plugins doctor voxtral-voice-call
```
