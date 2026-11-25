"""Normalize input media and read sample-accurate audio chunks."""

from __future__ import annotations

import math
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, TYPE_CHECKING

from . import checkpoints, runtime

if TYPE_CHECKING:
    import numpy as np


SAMPLE_RATE = 16_000


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required executable not found in PATH: {name}")
    return path


def ensure_normalized_wav(input_path: Path, work_dir: Path, force: bool = False) -> Path:
    """Create/reuse a mono 16 kHz PCM16 WAV. This makes chunk boundaries sample-accurate."""
    ffmpeg = require_binary("ffmpeg")

    work_dir.mkdir(parents=True, exist_ok=True)
    wav_path = work_dir / "audio_16k_mono.wav"
    meta_path = work_dir / "audio_16k_mono.meta.json"

    source_id = checkpoints.file_identity(input_path)
    expected_meta = {
        # A legacy entry may already pair a new WAV with old source metadata.
        "cache_version": 1,
        "source": source_id,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "sample_width_bytes": 2,
    }

    if not force and wav_path.is_file() and meta_path.is_file():
        try:
            existing = checkpoints.load_json(meta_path)
            if existing == expected_meta:
                validate_normalized_wav(wav_path)
                print(f"Reusing normalized audio: {wav_path}")
                return wav_path
        except Exception:
            pass

    print("Normalizing audio once to mono 16 kHz PCM WAV...")
    tmp = wav_path.with_suffix(".wav.tmp")
    if tmp.exists():
        tmp.unlink()

    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-protocol_whitelist",
        "file,crypto,data",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(tmp),
    ]

    try:
        subprocess.run(command, check=True)
        validate_normalized_wav(tmp)
        if checkpoints.file_identity(input_path) != source_id:
            raise RuntimeError(f"Input changed while FFmpeg was reading it: {input_path}")
        # Invalidate the old metadata before replacing its WAV. An interrupted
        # publish can then leave a cache miss, but never a mismatched cache hit.
        checkpoints.save_json_atomic(meta_path, None)
        tmp.replace(wav_path)
        checkpoints.save_json_atomic(meta_path, expected_meta)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

    print(f"Normalized audio saved: {wav_path}")
    return wav_path


def validate_normalized_wav(path: Path) -> None:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise RuntimeError(f"Expected mono WAV, got {wf.getnchannels()} channels")
        if wf.getframerate() != SAMPLE_RATE:
            raise RuntimeError(f"Expected {SAMPLE_RATE} Hz WAV, got {wf.getframerate()} Hz")
        if wf.getsampwidth() != 2:
            raise RuntimeError(f"Expected PCM16 WAV, sample width is {wf.getsampwidth()} bytes")
        if wf.getcomptype() != "NONE":
            raise RuntimeError(f"Expected uncompressed PCM WAV, got {wf.getcomptype()}")


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def read_wav_interval(path: Path, start: float, end: float) -> np.ndarray:
    """Read [start, end) exactly by PCM frame and return float32 mono audio in [-1, 1]."""
    if end <= start:
        return runtime.np.empty((0,), dtype=runtime.np.float32)

    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        total = wf.getnframes()
        start_frame = max(0, min(total, int(round(start * rate))))
        end_frame = max(start_frame, min(total, int(round(end * rate))))
        wf.setpos(start_frame)
        raw = wf.readframes(end_frame - start_frame)

    pcm = runtime.np.frombuffer(raw, dtype="<i2")
    return pcm.astype(runtime.np.float32) / 32768.0


def build_chunks(duration: float, chunk_seconds: float, context_seconds: float) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    count = max(1, math.ceil(duration / chunk_seconds))

    for index in range(count):
        core_start = index * chunk_seconds
        core_end = min(duration, core_start + chunk_seconds)
        ext_start = max(0.0, core_start - context_seconds)
        ext_end = min(duration, core_end + context_seconds)
        chunks.append(
            {
                "index": index,
                "core_start": core_start,
                "core_end": core_end,
                "ext_start": ext_start,
                "ext_end": ext_end,
                "is_last": index == count - 1,
            }
        )

    return chunks


def in_core(midpoint: float, chunk: dict[str, Any]) -> bool:
    if chunk["is_last"]:
        return chunk["core_start"] <= midpoint <= chunk["core_end"] + 1e-6
    return chunk["core_start"] <= midpoint < chunk["core_end"]
