"""Pure-stdlib audio glue: G.711 mu-law <-> PCM16, linear resampling, float32 -> int16, WAV reading."""

from __future__ import annotations

import array
import struct
import wave
from pathlib import Path
from typing import Optional, Tuple

_ULAW_SEG_END = (0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF)


def _ulaw2linear(u: int) -> int:
    u = ~u & 0xFF
    t = ((u & 0x0F) << 3) + 0x84
    t <<= (u & 0x70) >> 4
    return (0x84 - t) if (u & 0x80) else (t - 0x84)


def _linear2ulaw(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0x00
    if sample < 0:
        sample = -sample
    if sample > 32635:
        sample = 32635
    sample += 0x84
    seg = 0
    for seg in range(8):
        if sample <= _ULAW_SEG_END[seg]:
            break
    else:
        return (~(0x7F ^ sign)) & 0xFF
    uval = sign | (seg << 4) | ((sample >> (seg + 3)) & 0x0F)
    return (~uval) & 0xFF


def mulaw_decode(data: bytes) -> bytes:
    """mu-law bytes -> 16-bit little-endian mono PCM."""
    out = array.array("h", [_ulaw2linear(b) for b in data])
    return out.tobytes()


def mulaw_encode(pcm16: bytes) -> bytes:
    """16-bit little-endian mono PCM -> mu-law bytes."""
    samples = array.array("h")
    samples.frombytes(pcm16)
    return bytes(_linear2ulaw(s) for s in samples)


def resample_pcm16(data: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resample of 16-bit mono PCM."""
    if src_rate == dst_rate or not data:
        return data
    samples = array.array("h")
    samples.frombytes(data)
    n = len(samples)
    if n < 2:
        return data
    out_len = max(1, int(round(n * dst_rate / src_rate)))
    step = (n - 1) / max(out_len - 1, 1)
    out = array.array("h", bytes(2 * out_len))
    for i in range(out_len):
        pos = i * step
        i0 = int(pos)
        frac = pos - i0
        i1 = min(i0 + 1, n - 1)
        out[i] = int(samples[i0] * (1.0 - frac) + samples[i1] * frac)
    return out.tobytes()


def float32_to_pcm16(data: bytes) -> bytes:
    """Little-endian float32 samples -> clamped 16-bit little-endian PCM."""
    n = len(data) // 4
    if n == 0:
        return b""
    floats = struct.unpack(f"<{n}f", data[: n * 4])
    out = array.array("h", (max(-32768, min(32767, int(f * 32767.0))) for f in floats))
    return out.tobytes()


def read_wav(path: str) -> Tuple[Optional[bytes], int]:
    """Read a PCM WAV file -> (pcm16 mono bytes, sample rate); None when unreadable or not mono 16-bit."""
    try:
        with wave.open(str(Path(path)), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                return None, 0
            rate = wav.getframerate()
            return wav.readframes(wav.getnframes()), rate
    except Exception:
        return None, 0


def chunk_bytes(data: bytes, size: int):
    for i in range(0, len(data), size):
        yield data[i : i + size]
