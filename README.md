# voxtral-voice-call — call Hermes and talk to it fluidly

A standalone Hermes plugin that gives Hermes a **Voxtral 3 voice on every voice-capable
surface — no Twilio needed**:

- **STT provider `voxtral-realtime`** (`stt.provider`) — batch transcription for voice notes
  and Discord-voice utterances, plus live streaming sessions (`stt.streaming: true`) for
  Desktop/CLI voice chat: Voxtral Mini Transcribe Realtime 3 over websocket, 16 kHz PCM in,
  partial transcripts out.
- **TTS provider `voxtral-realtime`** (`tts.provider`) — Voxtral Mini TTS 3 over a **pool of
  persistent realtime websockets** (per model/voice), so replies start ~90 ms after each
  sentence is generated — gpt-live-style, no per-sentence reconnect. `streams_pcm` opts into
  the gateway's streaming-TTS consumer, covering Discord voice replies, Desktop speak-stream,
  and voice bubbles.
- **Optional phone-call adapter** (`voice-call` platform, only active when `TWILIO_*` env is
  set) — answers calls on a Twilio number: caller audio streams to Voxtral Mini Transcribe
  Realtime 3 with server-side turn detection, each turn runs through the Hermes agent on a
  per-call session, and the reply is spoken back with barge-in support.

No Hermes core changes — everything ships as this plugin: two providers plus an optional
gateway platform adapter.

## Use on Discord voice or Desktop voice chat (no Twilio)

```yaml
# config.yaml
stt:
  provider: voxtral-realtime
  streaming: true          # live 16 kHz sessions for voice chat (Desktop/CLI chained mode)
tts:
  provider: voxtral-realtime
voice:
  auto_tts: true           # speak replies aloud on voice turns
  silence_duration: 1.2    # snappier turn-end (default 3.0)
```

- **Discord**: `/voice join` in a voice channel — utterances are transcribed by
  `voxtral-realtime` and replies are spoken into the channel via the gateway's streaming-TTS
  contract (barge-in supported).
- **Desktop / CLI voice chat**: chained voice mode (`voice.voice_chat_mode: chained`, the
  default) picks up the same two providers — mic → live STT → Hermes agent → sentence-streamed
  TTS, with playback barge-in. `gpt-live` mode is OpenAI-only; not needed here.
- **Any other gateway with voice** (voice messages, voice bubbles): `stt.provider` +
  `tts.provider` cover them.

`~/.hermes/.env` only needs:

```dotenv
MISTRAL_API_KEY=...   # template already in ~/.hermes/.env; console.mistral.ai
```

All plugin settings (models, voice, STT delay/language/bias, webhook, allowlist, consent, call
cap, Twilio creds) are also editable in the **plugin settings UI** — Desktop → Settings →
Plugins → `voxtral-voice-call` card (rendered from the manifest `config_schema`; secrets go to
`~/.hermes/.env`, the rest to `plugins.entries.voxtral-voice-call.settings` in config.yaml).

### Switch the voice

Pick any Voxtral 3 preset in the Desktop TTS voice dropdown, or from a terminal:

```bash
hermes config set tts.voice gb_jane_neutral     # e.g. en_paul_happy, fr_marie_neutral, voice_3150 (James, en-US), ...
```

The TTS connection pool keeps one persistent websocket per voice, so switching is instant — no
reconnect penalty. Full preset list: `VOXTRAL3_VOICES` in `tts_provider.py`.

### Logs

Plugin logs under the `voxtral-voice-call` logger (WS connect/session, per-sentence chunk
counts, error events with full payload, HTTP codes — never the API key). After reproducing an
issue:

```bash
grep -i voxtral ~/.hermes/logs/gui.log ~/.hermes/logs/gateway.log ~/.hermes/logs/agent.log ~/.hermes/logs/errors.log | tail -100
```

## Optional: real phone calls via Twilio

If you later want an actual phone number, add the `TWILIO_*` vars below and enable
`platforms.voice-call` — see "Configuration".



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

## Configuration (Twilio phone-call adapter — optional)

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
- Realtime wire field names follow the Voxtral 3 public-preview protocol as pinned by the
  official SDK / LiveKit plugin; if Mistral renames fields at GA, only `voxtral.py` changes.

## Development

```bash
python -m py_compile *.py
pytest tests/
hermes plugins validate /path/to/voxtral-voice-call   # from a hermes-agent checkout
hermes plugins doctor voxtral-voice-call
```
