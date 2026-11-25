from __future__ import annotations

# This program is intentionally offline during normal operation.
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"

import argparse
import gc
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unicodedata
import uuid
import wave
from pathlib import Path
from typing import Any

from job_state import JOB_STATE_NAME, JobBusyError, atomic_text_file, job_lock, publish_exports
from local_models import file_sha256, missing_pyannote_assets, whisper_model_error

# Loaded only after CLI parsing and local-model validation. This keeps --help fast.
np: Any = None
torch: Any = None
whisper: Any = None


# -----------------------------------------------------------------------------
# Local model layout
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
WHISPER_MODEL_PATH = MODELS_DIR / "whisper" / "large-v3-turbo.pt"
PYANNOTE_MODEL_PATH = MODELS_DIR / "pyannote" / "speaker-diarization-community-1"

SAMPLE_RATE = 16_000
CHECKPOINT_SCHEMA_VERSION = 2
BOUNDARY_MATCH_SECONDS = 0.75
BOUNDARY_RECOVERY_SECONDS = 0.75


def load_runtime_dependencies() -> None:
    global np, torch, whisper
    import numpy as numpy_module
    import torch as torch_module
    import whisper as whisper_module

    np = numpy_module
    torch = torch_module
    whisper = whisper_module


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json_atomic(path: Path, data: Any) -> None:
    with atomic_text_file(path) as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_default)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def directory_identity(path: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        try:
            rel = item.relative_to(path)
        except ValueError:
            continue
        if ".git" in rel.parts:
            continue
        stat = item.stat()
        files.append(
            {
                "path": str(rel).replace("\\", "/"),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {"path": str(path.resolve()), "files": files}


def validate_local_models(*, require_pyannote: bool = True) -> None:
    errors = []
    setup_flag = "" if require_pyannote else " --whisper-only"
    setup = f"`bash setup_models.sh{setup_flag}` or `setup_models.cmd{setup_flag}` on Windows"

    whisper_error = whisper_model_error(WHISPER_MODEL_PATH)
    if whisper_error:
        errors.append(f"Whisper Turbo model {whisper_error}: {WHISPER_MODEL_PATH}")

    if require_pyannote:
        missing = missing_pyannote_assets(PYANNOTE_MODEL_PATH)
        if missing:
            assets = ", ".join(missing)
            errors.append(f"pyannote Community-1 model missing/incomplete: {PYANNOTE_MODEL_PATH} ({assets})")

    if errors:
        errors.append(f"Run {setup} to install or repair the local models.")
        raise RuntimeError("\n\n".join(errors))


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required executable not found in PATH: {name}")
    return path


def clear_gpu() -> None:
    if torch is None:
        return
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


def print_gpu_memory(prefix: str = "") -> None:
    if torch is None or not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"{prefix}CUDA: {allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved")


def format_timestamp(seconds: float, srt: bool = False) -> str:
    ms = max(0, int(round(seconds * 1000)))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    sep = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


# -----------------------------------------------------------------------------
# Audio normalization and exact chunk reading
# -----------------------------------------------------------------------------

def ensure_normalized_wav(input_path: Path, work_dir: Path, force: bool = False) -> Path:
    """Create/reuse a mono 16 kHz PCM16 WAV. This makes chunk boundaries sample-accurate."""
    ffmpeg = require_binary("ffmpeg")

    work_dir.mkdir(parents=True, exist_ok=True)
    wav_path = work_dir / "audio_16k_mono.wav"
    meta_path = work_dir / "audio_16k_mono.meta.json"

    source_id = file_identity(input_path)
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
            existing = load_json(meta_path)
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
        if file_identity(input_path) != source_id:
            raise RuntimeError(f"Input changed while FFmpeg was reading it: {input_path}")
        # Invalidate the old metadata before replacing its WAV. An interrupted
        # publish can then leave a cache miss, but never a mismatched cache hit.
        save_json_atomic(meta_path, None)
        tmp.replace(wav_path)
        save_json_atomic(meta_path, expected_meta)
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
        return np.empty((0,), dtype=np.float32)

    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        total = wf.getnframes()
        start_frame = max(0, min(total, int(round(start * rate))))
        end_frame = max(start_frame, min(total, int(round(end * rate))))
        wf.setpos(start_frame)
        raw = wf.readframes(end_frame - start_frame)

    pcm = np.frombuffer(raw, dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


# -----------------------------------------------------------------------------
# Chunk layout
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Stage 1: Whisper (all chunks, then unload)
# -----------------------------------------------------------------------------

def whisper_stage_signature(
    normalized_wav: Path,
    language: str | None,
    prompt: str | None,
    chunk_seconds: float,
    context_seconds: float,
    *,
    audio_sha256: str | None = None,
) -> str:
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "normalized_audio_sha256": audio_sha256 or file_sha256(normalized_wav),
        "model": file_identity(WHISPER_MODEL_PATH),
        "language": language,
        "prompt": prompt,
        "chunk_seconds": chunk_seconds,
        "context_seconds": context_seconds,
        "word_timestamps": True,
        "alignment_heads": "turbo",
        "beam_size": 5,
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "condition_on_previous_text": True,
        "hallucination_silence_threshold": 2.0,
    }
    return stable_hash(payload)


def finite_number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def valid_timed_item(item: Any, *, speaker: bool = False) -> bool:
    if not isinstance(item, dict):
        return False
    start = item.get("start")
    end = item.get("end")
    if not (finite_number(start) and finite_number(end) and 0 <= start <= end):
        return False
    key = "speaker" if speaker else "text"
    if not isinstance(item.get(key), str):
        return False
    probability = item.get("probability")
    return speaker or probability is None or (
        finite_number(probability) and 0 <= probability <= 1
    )


def load_valid_chunk_checkpoint(
    path: Path,
    signature: str,
    chunk: dict[str, Any],
    stage: str,
    force_generation: str | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = load_json(path)
        if not isinstance(data, dict) or data.get("stage_signature") != signature:
            return None
        if data.get("chunk") != chunk or data.get("stage") != stage:
            return None
        if force_generation is not None and data.get("force_generation") != force_generation:
            return None
        if stage == "whisper":
            words = data.get("words_extended")
            if not isinstance(words, list) or not all(valid_timed_item(word) for word in words):
                return None
        elif stage == "diarization":
            turns = data.get("turns_extended")
            embeddings = data.get("embeddings")
            durations = data.get("core_durations")
            if not isinstance(turns, list) or not all(valid_timed_item(turn, speaker=True) for turn in turns):
                return None
            if not isinstance(embeddings, dict) or not all(
                isinstance(label, str)
                and isinstance(vector, list)
                and all(isinstance(value, (int, float)) for value in vector)
                for label, vector in embeddings.items()
            ):
                return None
            if not isinstance(durations, dict) or not all(
                isinstance(label, str)
                and finite_number(value)
                and value >= 0
                for label, value in durations.items()
            ):
                return None
        else:
            return None
        return data
    except Exception:
        return None


def prepare_force_run(
    output_dir: Path,
    force_whisper: bool,
    force_diarization: bool,
    whisper_signature: str,
    diarization_signature: str,
) -> tuple[str | None, str | None, Path | None]:
    """Resume an interrupted forced pass; start a new one after a completed pass."""
    manifest_path = output_dir / "work" / "force_run.json"
    requested = {
        "whisper": whisper_signature if force_whisper else None,
        "diarization": diarization_signature if force_diarization else None,
    }
    active: dict[str, Any] | None = None
    if manifest_path.is_file():
        try:
            candidate = load_json(manifest_path)
            if isinstance(candidate, dict) and candidate.get("complete") is False:
                generation = candidate.get("generation")
                stages = candidate.get("requested")
                if (
                    isinstance(generation, str)
                    and generation
                    and isinstance(stages, dict)
                    and stages.keys() == requested.keys()
                    and all(value is None or isinstance(value, str) and value for value in stages.values())
                    and any(stages.values())
                ):
                    active = candidate
        except (OSError, ValueError):
            pass

    if force_whisper or force_diarization:
        if active is None or active.get("requested") != requested:
            active = {"generation": uuid.uuid4().hex, "requested": requested, "complete": False}
            save_json_atomic(manifest_path, active)
    elif active is not None:
        previous = active.get("requested", {})
        if (
            previous.get("whisper") not in (None, whisper_signature)
            or previous.get("diarization") not in (None, diarization_signature)
        ):
            active = None

    if active is None:
        return None, None, None
    requested = active["requested"]
    generation = active["generation"]
    return (
        generation if requested.get("whisper") is not None else None,
        generation if requested.get("diarization") is not None else None,
        manifest_path,
    )


def complete_force_run(manifest_path: Path | None) -> None:
    if manifest_path is None:
        return
    manifest = load_json(manifest_path)
    manifest["complete"] = True
    save_json_atomic(manifest_path, manifest)


def load_local_whisper(device: str) -> Any:
    model = whisper.load_model(str(WHISPER_MODEL_PATH), device=device)
    # Upstream applies these heads for the model name, but skips them for a path.
    model.set_alignment_heads(whisper._ALIGNMENT_HEADS["turbo"])
    model.alignment_heads = model.alignment_heads.to(device)
    return model


def run_whisper_chunks(
    normalized_wav: Path,
    chunks: list[dict[str, Any]],
    checkpoint_dir: Path,
    language: str | None,
    prompt: str | None,
    chunk_seconds: float,
    context_seconds: float,
    force_generation: str | None,
    verbose: bool,
    *,
    stage_signature: str | None = None,
) -> None:
    signature = stage_signature or whisper_stage_signature(
        normalized_wav, language, prompt, chunk_seconds, context_seconds
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for chunk in chunks:
        path = checkpoint_dir / f"chunk_{chunk['index']:06d}.json"
        if load_valid_chunk_checkpoint(path, signature, chunk, "whisper", force_generation) is None:
            pending.append(chunk)

    print("\n" + "=" * 80)
    print("STAGE 1 / 3 - WHISPER TURBO")
    print("=" * 80)
    print(f"Chunks total: {len(chunks)} | pending: {len(pending)}")

    if not pending:
        print("All Whisper chunks already checkpointed. Skipping model load.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading local Whisper model: {WHISPER_MODEL_PATH}")
    print(f"Device: {device}")

    model = load_local_whisper(device)
    print_gpu_memory("After Whisper load: ")

    try:
        for n, chunk in enumerate(chunks, start=1):
            checkpoint = checkpoint_dir / f"chunk_{chunk['index']:06d}.json"

            if load_valid_chunk_checkpoint(checkpoint, signature, chunk, "whisper", force_generation):
                print(f"[Whisper {n}/{len(chunks)}] checkpoint exists -> skip")
                continue

            print(
                f"\n[Whisper {n}/{len(chunks)}] "
                f"core {format_timestamp(chunk['core_start'])} -> {format_timestamp(chunk['core_end'])} | "
                f"read {format_timestamp(chunk['ext_start'])} -> {format_timestamp(chunk['ext_end'])}"
            )

            audio = read_wav_interval(normalized_wav, chunk["ext_start"], chunk["ext_end"])

            result = model.transcribe(
                audio,
                task="transcribe",
                language=language,
                word_timestamps=True,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                beam_size=5,
                condition_on_previous_text=True,
                initial_prompt=prompt,
                hallucination_silence_threshold=2.0,
                fp16=(device == "cuda"),
                verbose=verbose,
            )

            words_extended: list[dict[str, Any]] = []
            ext_start = chunk["ext_start"]

            for segment in result.get("segments", []):
                for word in segment.get("words", []):
                    if word.get("start") is None or word.get("end") is None:
                        continue
                    start = ext_start + float(word["start"])
                    end = ext_start + float(word["end"])
                    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
                        continue
                    probability = word.get("probability")
                    probability = float(probability) if probability is not None else None
                    if probability is not None and not math.isfinite(probability):
                        probability = None
                    words_extended.append(
                        {
                            "start": start,
                            "end": end,
                            "text": str(word.get("word", "")),
                            "probability": probability,
                        }
                    )

            checkpoint_data = {
                "stage": "whisper",
                "stage_signature": signature,
                "force_generation": force_generation,
                "chunk": chunk,
                "detected_language": result.get("language"),
                "words_extended": words_extended,
            }
            save_json_atomic(checkpoint, checkpoint_data)
            print(f"Saved {len(words_extended)} extended words -> {checkpoint}")

            del result
            del audio
            gc.collect()

    finally:
        del model
        clear_gpu()
        print_gpu_memory("After Whisper unload: ")


# -----------------------------------------------------------------------------
# Stage 2: pyannote Community-1 (all chunks, then unload)
# -----------------------------------------------------------------------------

def pyannote_stage_signature(
    normalized_wav: Path,
    min_speakers: int | None,
    max_speakers: int | None,
    chunk_seconds: float,
    context_seconds: float,
    *,
    audio_sha256: str | None = None,
) -> str:
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "normalized_audio_sha256": audio_sha256 or file_sha256(normalized_wav),
        "model": directory_identity(PYANNOTE_MODEL_PATH),
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "chunk_seconds": chunk_seconds,
        "context_seconds": context_seconds,
        "exclusive_diarization": True,
    }
    return stable_hash(payload)


def annotation_to_turns(annotation: Any, offset: float) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []

    if hasattr(annotation, "itertracks"):
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                {
                    "start": offset + float(turn.start),
                    "end": offset + float(turn.end),
                    "speaker": str(speaker),
                }
            )
    else:
        for item in annotation:
            if len(item) == 2:
                turn, speaker = item
            elif len(item) == 3:
                turn, _, speaker = item
            else:
                continue
            turns.append(
                {
                    "start": offset + float(turn.start),
                    "end": offset + float(turn.end),
                    "speaker": str(speaker),
                }
            )

    turns.sort(key=lambda x: (x["start"], x["end"], x["speaker"]))
    return turns


def clipped_duration_for_speaker(
    turns: list[dict[str, Any]], speaker: str, start: float, end: float
) -> float:
    total = 0.0
    for turn in turns:
        if turn["speaker"] != speaker:
            continue
        total += max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
    return total


def run_pyannote_chunks(
    normalized_wav: Path,
    chunks: list[dict[str, Any]],
    checkpoint_dir: Path,
    min_speakers: int | None,
    max_speakers: int | None,
    chunk_seconds: float,
    context_seconds: float,
    force_generation: str | None,
    *,
    stage_signature: str | None = None,
) -> None:
    signature = stage_signature or pyannote_stage_signature(
        normalized_wav, min_speakers, max_speakers, chunk_seconds, context_seconds
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for chunk in chunks:
        path = checkpoint_dir / f"chunk_{chunk['index']:06d}.json"
        if load_valid_chunk_checkpoint(path, signature, chunk, "diarization", force_generation) is None:
            pending.append(chunk)

    print("\n" + "=" * 80)
    print("STAGE 2 / 3 - PYANNOTE COMMUNITY-1")
    print("=" * 80)
    print(f"Chunks total: {len(chunks)} | pending: {len(pending)}")

    if not pending:
        print("All diarization chunks already checkpointed. Skipping model load.")
        return

    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading local pyannote model: {PYANNOTE_MODEL_PATH}")
    print(f"Device: {device}")

    pipeline = Pipeline.from_pretrained(str(PYANNOTE_MODEL_PATH))
    pipeline.to(device)
    print_gpu_memory("After pyannote load: ")

    try:
        for n, chunk in enumerate(chunks, start=1):
            checkpoint = checkpoint_dir / f"chunk_{chunk['index']:06d}.json"

            if load_valid_chunk_checkpoint(checkpoint, signature, chunk, "diarization", force_generation):
                print(f"[pyannote {n}/{len(chunks)}] checkpoint exists -> skip")
                continue

            print(
                f"\n[pyannote {n}/{len(chunks)}] "
                f"core {format_timestamp(chunk['core_start'])} -> {format_timestamp(chunk['core_end'])} | "
                f"read {format_timestamp(chunk['ext_start'])} -> {format_timestamp(chunk['ext_end'])}"
            )

            audio = read_wav_interval(normalized_wav, chunk["ext_start"], chunk["ext_end"])
            waveform = torch.from_numpy(audio).unsqueeze(0)

            kwargs: dict[str, Any] = {}
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers

            with ProgressHook() as hook:
                output = pipeline(
                    {"waveform": waveform, "sample_rate": SAMPLE_RATE},
                    hook=hook,
                    **kwargs,
                )

            regular = output.speaker_diarization
            exclusive = output.exclusive_speaker_diarization
            labels = [str(label) for label in regular.labels()]
            embeddings_array = np.asarray(output.speaker_embeddings, dtype=np.float32)

            if embeddings_array.shape[0] != len(labels):
                raise RuntimeError(
                    "pyannote returned an unexpected speaker_embeddings shape: "
                    f"{embeddings_array.shape} for {len(labels)} labels"
                )

            embeddings: dict[str, list[float]] = {
                label: embeddings_array[i].tolist() for i, label in enumerate(labels)
            }

            turns = annotation_to_turns(exclusive, chunk["ext_start"])
            core_durations = {
                label: clipped_duration_for_speaker(
                    turns, label, chunk["core_start"], chunk["core_end"]
                )
                for label in labels
            }

            checkpoint_data = {
                "stage": "diarization",
                "stage_signature": signature,
                "force_generation": force_generation,
                "chunk": chunk,
                "turns_extended": turns,
                "embeddings": embeddings,
                "core_durations": core_durations,
            }
            save_json_atomic(checkpoint, checkpoint_data)
            print(
                f"Saved {len(turns)} exclusive turns / {len(labels)} local speakers -> {checkpoint}"
            )

            del output
            del waveform
            del audio
            del embeddings_array
            gc.collect()

    finally:
        del pipeline
        clear_gpu()
        print_gpu_memory("After pyannote unload: ")


# -----------------------------------------------------------------------------
# Stage 3A: reconcile local pyannote labels into stable global speaker IDs
# -----------------------------------------------------------------------------

def normalize_vector(v: np.ndarray) -> np.ndarray | None:
    if v.ndim != 1 or not np.all(np.isfinite(v)):
        return None
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return None
    return v / norm


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return -1.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def total_turn_overlap(
    current_turns: list[dict[str, Any]],
    current_local: str,
    previous_turns: list[dict[str, Any]],
    previous_global: str,
) -> float:
    a = [t for t in current_turns if t["speaker"] == current_local]
    b = [t for t in previous_turns if t["speaker"] == previous_global]
    total = 0.0
    i = 0
    j = 0

    while i < len(a) and j < len(b):
        left = max(a[i]["start"], b[j]["start"])
        right = min(a[i]["end"], b[j]["end"])
        if right > left:
            total += right - left

        if a[i]["end"] <= b[j]["end"]:
            i += 1
        else:
            j += 1

    return total


def maximum_weight_pairs(candidates: list[tuple[float, str, str]]) -> list[tuple[float, str, str]]:
    """Find the best one-to-one assignment, allowing either side to stay unmatched."""
    if not candidates:
        return []
    local_labels = sorted({local for _, local, _ in candidates})
    global_labels = sorted({global_id for _, _, global_id in candidates})
    weights = {(local, global_id): score for score, local, global_id in candidates}
    row_count = len(local_labels)
    real_columns = len(global_labels)
    column_count = real_columns + row_count  # zero-weight dummy columns
    maximum = max(score for score, _, _ in candidates)
    costs = [
        [maximum - weights.get((local, global_id), 0.0) for global_id in global_labels]
        + [maximum] * row_count
        for local in local_labels
    ]

    # Rectangular Hungarian assignment: row_count <= column_count.
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    column_row = [0] * (column_count + 1)
    predecessor = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        column_row[0] = row
        column = 0
        minimum = [float("inf")] * (column_count + 1)
        used = [False] * (column_count + 1)
        while True:
            used[column] = True
            current_row = column_row[column]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                reduced = (
                    costs[current_row - 1][candidate_column - 1]
                    - row_potential[current_row]
                    - column_potential[candidate_column]
                )
                if reduced < minimum[candidate_column]:
                    minimum[candidate_column] = reduced
                    predecessor[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    row_potential[column_row[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if column_row[column] == 0:
                break
        while column:
            previous_column = predecessor[column]
            column_row[column] = column_row[previous_column]
            column = previous_column

    matches = []
    for column in range(1, real_columns + 1):
        row = column_row[column]
        if row:
            local = local_labels[row - 1]
            global_id = global_labels[column - 1]
            score = weights.get((local, global_id))
            if score is not None:
                matches.append((score, local, global_id))
    return matches


def clip_turns_to_core(
    turns: list[dict[str, Any]],
    mapping: dict[str, str],
    core_start: float,
    core_end: float,
) -> list[dict[str, Any]]:
    clipped: list[dict[str, Any]] = []
    for turn in turns:
        start = max(core_start, float(turn["start"]))
        end = min(core_end, float(turn["end"]))
        if end - start <= 1e-4:
            continue
        local = turn["speaker"]
        clipped.append(
            {
                "start": start,
                "end": end,
                "speaker": mapping.get(local, "UNKNOWN"),
            }
        )
    return clipped


def merge_adjacent_turns(turns: list[dict[str, Any]], gap_tolerance: float = 0.05) -> list[dict[str, Any]]:
    if not turns:
        return []

    turns = sorted(turns, key=lambda x: (x["start"], x["end"]))
    merged = [dict(turns[0])]

    for turn in turns[1:]:
        prev = merged[-1]
        if (
            turn["speaker"] == prev["speaker"]
            and turn["start"] <= prev["end"] + gap_tolerance
        ):
            prev["end"] = max(prev["end"], turn["end"])
        else:
            merged.append(dict(turn))

    return merged


def reconcile_speakers(
    chunks: list[dict[str, Any]],
    checkpoint_dir: Path,
    output_path: Path,
    embedding_threshold: float,
    overlap_match_seconds: float,
) -> list[dict[str, Any]]:
    print("\n" + "=" * 80)
    print("STAGE 3 / 3 - SPEAKER RECONCILIATION + MERGE")
    print("=" * 80)

    global_profiles: dict[str, dict[str, Any]] = {}
    next_global_id = 0
    previous_mapped_extended: list[dict[str, Any]] = []
    final_core_turns: list[dict[str, Any]] = []
    report_chunks: list[dict[str, Any]] = []

    def new_global_id() -> str:
        nonlocal next_global_id
        value = f"SPEAKER_{next_global_id:02d}"
        next_global_id += 1
        return value

    for chunk in chunks:
        checkpoint = checkpoint_dir / f"chunk_{chunk['index']:06d}.json"
        data = load_json(checkpoint)
        turns = data["turns_extended"]
        raw_embeddings: dict[str, list[float]] = data.get("embeddings", {})
        core_durations: dict[str, float] = data.get("core_durations", {})

        local_labels = sorted(
            set(raw_embeddings.keys()) | {str(t["speaker"]) for t in turns}
        )
        local_vectors = {
            label: normalize_vector(np.asarray(raw_embeddings[label], dtype=np.float32))
            if label in raw_embeddings
            else None
            for label in local_labels
        }

        mapping: dict[str, str] = {}
        evidence: dict[str, dict[str, Any]] = {}
        used_globals: set[str] = set()

        # 1) Strongest evidence: both chunks contain the same overlap audio.
        overlap_candidates: list[tuple[float, str, str]] = []
        if previous_mapped_extended:
            previous_global_labels = sorted({t["speaker"] for t in previous_mapped_extended})
            for local in local_labels:
                for global_id in previous_global_labels:
                    seconds = total_turn_overlap(turns, local, previous_mapped_extended, global_id)
                    if seconds >= overlap_match_seconds:
                        overlap_candidates.append((seconds, local, global_id))

        for seconds, local, global_id in maximum_weight_pairs(overlap_candidates):
            mapping[local] = global_id
            used_globals.add(global_id)
            evidence[local] = {"method": "shared_audio_overlap", "seconds": seconds}

        # 2) Returning speakers that were not present in the overlap: embedding match.
        embedding_candidates: list[tuple[float, str, str]] = []
        for local in local_labels:
            if local in mapping:
                continue
            vector = local_vectors[local]
            if vector is None:
                continue
            for global_id, profile in global_profiles.items():
                if global_id in used_globals:
                    continue
                similarity = cosine_similarity(vector, profile["centroid"])
                if similarity >= embedding_threshold:
                    embedding_candidates.append((similarity, local, global_id))

        for similarity, local, global_id in maximum_weight_pairs(embedding_candidates):
            mapping[local] = global_id
            used_globals.add(global_id)
            evidence[local] = {"method": "embedding", "cosine_similarity": similarity}

        # 3) Anything still unmatched becomes a new stable speaker.
        for local in local_labels:
            if local in mapping:
                continue
            global_id = new_global_id()
            mapping[local] = global_id
            used_globals.add(global_id)
            evidence[local] = {"method": "new_speaker"}

        # Update global embedding profiles. Weight by committed core speech duration.
        for local, global_id in mapping.items():
            vector = local_vectors.get(local)
            if vector is None:
                continue

            weight = float(core_durations.get(local, 0.0))
            if weight <= 0.0:
                # Context-only appearance: useful, but do not let it dominate a profile.
                weight = 0.25

            if global_id not in global_profiles:
                global_profiles[global_id] = {
                    "weighted_sum": vector.astype(np.float64) * weight,
                    "weight": weight,
                    "centroid": vector.astype(np.float64),
                }
            else:
                profile = global_profiles[global_id]
                profile["weighted_sum"] += vector.astype(np.float64) * weight
                profile["weight"] += weight
                centroid = normalize_vector(profile["weighted_sum"].astype(np.float32))
                if centroid is not None:
                    profile["centroid"] = centroid.astype(np.float64)

        mapped_extended = [
            {
                "start": float(turn["start"]),
                "end": float(turn["end"]),
                "speaker": mapping.get(str(turn["speaker"]), "UNKNOWN"),
            }
            for turn in turns
        ]
        previous_mapped_extended = mapped_extended

        core_turns = clip_turns_to_core(
            turns,
            mapping,
            chunk["core_start"],
            chunk["core_end"],
        )
        final_core_turns.extend(core_turns)

        report_chunks.append(
            {
                "chunk_index": chunk["index"],
                "mapping": mapping,
                "evidence": evidence,
            }
        )

        readable = ", ".join(f"{local}->{global_id}" for local, global_id in mapping.items())
        print(f"Chunk {chunk['index']:06d}: {readable or 'no speech'}")

    final_core_turns = merge_adjacent_turns(final_core_turns)

    report = {
        "embedding_match_threshold": embedding_threshold,
        "overlap_match_seconds": overlap_match_seconds,
        "global_speakers": sorted(global_profiles.keys()),
        "chunks": report_chunks,
        "speaker_turns": final_core_turns,
    }
    save_json_atomic(output_path, report)
    print(f"Stable speakers: {len(global_profiles)}")
    print(f"Speaker reconciliation report: {output_path}")
    return final_core_turns


def save_single_speaker_reconciliation(path: Path) -> None:
    """Record the explicit user assumption without inventing diarization turns."""
    save_json_atomic(path, {
        "mode": "single_speaker",
        "global_speakers": ["SPEAKER_00"],
        "chunks": [],
        "speaker_turns": [],
    })


# -----------------------------------------------------------------------------
# Stage 3B: load Whisper words, assign speakers, create readable utterances
# -----------------------------------------------------------------------------

def word_midpoint(word: dict[str, Any]) -> float:
    return (float(word["start"]) + float(word["end"])) / 2.0


def matching_word_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip().casefold()
    return "".join(char for char in normalized if not unicodedata.category(char).startswith("P"))


def choose_boundary_word(
    previous: tuple[int, dict[str, Any]],
    current: tuple[int, dict[str, Any]],
    previous_chunk: dict[str, Any],
    current_chunk: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    previous_core = in_core(word_midpoint(previous[1]), previous_chunk)
    current_core = in_core(word_midpoint(current[1]), current_chunk)
    if previous_core != current_core:
        return previous if previous_core else current
    previous_probability = previous[1].get("probability")
    current_probability = current[1].get("probability")
    if previous_probability is not None and current_probability is not None:
        return previous if previous_probability >= current_probability else current
    return previous


def load_all_words(chunks: list[dict[str, Any]], checkpoint_dir: Path) -> list[dict[str, Any]]:
    """Merge independently decoded chunk edges without dropping or repeating matched words."""
    extended: list[list[tuple[int, dict[str, Any]]]] = []
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for chunk in chunks:
        path = checkpoint_dir / f"chunk_{chunk['index']:06d}.json"
        data = load_json(path)
        chunk_words = data["words_extended"]
        tagged = [(position, word) for position, word in enumerate(chunk_words)]
        extended.append(tagged)
        for position, word in tagged:
            if in_core(word_midpoint(word), chunk):
                selected[(chunk["index"], position)] = word

    removed = 0
    recovered = 0
    for position in range(len(chunks) - 1):
        previous_chunk, current_chunk = chunks[position:position + 2]
        if previous_chunk["ext_end"] <= current_chunk["ext_start"]:
            continue
        boundary = current_chunk["core_start"]
        window = min(
            2.0,
            previous_chunk["ext_end"] - boundary + 0.5,
            boundary - current_chunk["ext_start"] + 0.5,
        )
        previous_words = [
            item for item in extended[position]
            if abs(word_midpoint(item[1]) - boundary) <= window
        ]
        current_words = [
            item for item in extended[position + 1]
            if abs(word_midpoint(item[1]) - boundary) <= window
        ]
        candidates: list[tuple[float, int, int]] = []
        for previous_index, (_, previous_word) in enumerate(previous_words):
            previous_text = matching_word_text(previous_word["text"])
            if not previous_text:
                continue
            for current_index, (_, current_word) in enumerate(current_words):
                if previous_text != matching_word_text(current_word["text"]):
                    continue
                previous_midpoint = word_midpoint(previous_word)
                current_midpoint = word_midpoint(current_word)
                distance = abs(previous_midpoint - current_midpoint)
                overlap = max(
                    0.0,
                    min(previous_word["end"], current_word["end"])
                    - max(previous_word["start"], current_word["start"]),
                )
                shorter_duration = min(
                    previous_word["end"] - previous_word["start"],
                    current_word["end"] - current_word["start"],
                )
                if (
                    overlap == 0
                    and in_core(previous_midpoint, previous_chunk)
                    and in_core(current_midpoint, current_chunk)
                ):
                    continue
                if distance <= BOUNDARY_MATCH_SECONDS and (
                    distance <= 0.12
                    or (shorter_duration > 0 and overlap / shorter_duration >= 0.25)
                ):
                    candidates.append((distance, previous_index, current_index))

        matched_previous: set[int] = set()
        matched_current: set[int] = set()
        for _, previous_index, current_index in sorted(candidates):
            if previous_index in matched_previous or current_index in matched_current:
                continue
            matched_previous.add(previous_index)
            matched_current.add(current_index)
            previous = previous_words[previous_index]
            current = current_words[current_index]
            chosen = choose_boundary_word(previous, current, previous_chunk, current_chunk)
            previous_key = (previous_chunk["index"], previous[0])
            current_key = (current_chunk["index"], current[0])
            had_core_word = previous_key in selected or current_key in selected
            removed += int(previous_key in selected and current_key in selected)
            selected.pop(previous_key, None)
            selected.pop(current_key, None)
            if had_core_word or (
                abs(word_midpoint(previous[1]) - boundary) <= BOUNDARY_RECOVERY_SECONDS
                and abs(word_midpoint(current[1]) - boundary) <= BOUNDARY_RECOVERY_SECONDS
            ):
                chosen_chunk = previous_chunk if chosen is previous else current_chunk
                selected[(chosen_chunk["index"], chosen[0])] = chosen[1]
                recovered += int(not had_core_word)

    words = list(selected.values())
    words.sort(key=lambda word: (word["start"], word["end"]))
    if len(chunks) > 1:
        print(f"Whisper boundary merge: {removed} duplicate words removed, {recovered} words recovered")
    return words


def assign_speakers_to_words(
    words: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    gap_tolerance: float,
) -> list[dict[str, Any]]:
    if not words:
        return []
    if not turns:
        return [{**word, "speaker": "UNKNOWN"} for word in words]

    assigned: list[dict[str, Any]] = []
    pointer = 0
    n_turns = len(turns)

    for word in words:
        ws = float(word["start"])
        we = float(word["end"])
        midpoint = (ws + we) / 2.0

        while pointer + 1 < n_turns and turns[pointer]["end"] < ws - gap_tolerance:
            pointer += 1

        best_speaker = None
        best_overlap = 0.0
        best_distance = float("inf")

        j = max(0, pointer - 1)
        while j < n_turns and turns[j]["start"] <= we + gap_tolerance:
            turn = turns[j]
            overlap = max(0.0, min(we, turn["end"]) - max(ws, turn["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

            if turn["start"] <= midpoint <= turn["end"]:
                distance = 0.0
            else:
                distance = min(abs(midpoint - turn["start"]), abs(midpoint - turn["end"]))

            if best_overlap <= 0.0 and distance < best_distance:
                best_distance = distance
                if distance <= gap_tolerance:
                    best_speaker = turn["speaker"]
            j += 1

        assigned.append({**word, "speaker": best_speaker or "UNKNOWN"})

    return assigned


def needs_word_separator(previous: str, following: str) -> bool:
    if not previous or not following or previous[-1].isspace() or following[0].isspace():
        return False
    if following[0] in ".,!?;:%)]}、。，！？" or previous[-1] in "([{¿¡/-":
        return False
    if following.startswith("-") and len(following) > 1 and following[1].isalnum():
        return False
    if previous[-1] in "'’" and (len(previous) == 1 or not previous[-2].isalnum()):
        return False
    if following[0] in "'’" and following[1:].casefold() in {"s", "t", "re", "ve", "ll", "d", "m"}:
        return False
    # CJK scripts normally do not use spaces between written words.
    if any(
        "CJK" in unicodedata.name(char, "")
        or "HIRAGANA" in unicodedata.name(char, "")
        or "KATAKANA" in unicodedata.name(char, "")
        or "HANGUL" in unicodedata.name(char, "")
        for char in (previous[-1], following[0])
    ):
        return False
    return True


def append_word_part(parts: list[str], text: str) -> None:
    previous = next((part for part in reversed(parts) if part), "")
    parts.append((" " if needs_word_separator(previous, text) else "") + text)


def build_utterances(
    words: list[dict[str, Any]],
    max_duration: float,
    max_gap: float,
) -> list[dict[str, Any]]:
    if not words:
        return []

    utterances: list[dict[str, Any]] = []
    current = {
        "speaker": words[0]["speaker"],
        "start": float(words[0]["start"]),
        "end": float(words[0]["end"]),
        "parts": [str(words[0]["text"])],
    }

    for word in words[1:]:
        word_start = float(word["start"])
        word_end = float(word["end"])
        gap = word_start - float(current["end"])
        duration = word_end - float(current["start"])

        split = (
            word["speaker"] != current["speaker"]
            or gap > max_gap
            or duration > max_duration
        )

        if split:
            utterances.append(
                {
                    "speaker": current["speaker"],
                    "start": current["start"],
                    "end": current["end"],
                    "text": "".join(current["parts"]).strip(),
                }
            )
            current = {
                "speaker": word["speaker"],
                "start": word_start,
                "end": word_end,
                "parts": [str(word["text"])],
            }
        else:
            current["end"] = max(float(current["end"]), word_end)
            append_word_part(current["parts"], str(word["text"]))

    utterances.append(
        {
            "speaker": current["speaker"],
            "start": current["start"],
            "end": current["end"],
            "text": "".join(current["parts"]).strip(),
        }
    )

    return [u for u in utterances if u["text"]]


# -----------------------------------------------------------------------------
# Output writers
# -----------------------------------------------------------------------------

def save_txt(path: Path, utterances: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in utterances:
            f.write(
                f"[{format_timestamp(item['start'])} --> {format_timestamp(item['end'])}] "
                f"{item['speaker']}\n{item['text']}\n\n"
            )


def save_srt(path: Path, utterances: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for index, item in enumerate(utterances, start=1):
            text = f"{item['speaker']}: {item['text']}"
            wrapped = "\n".join(textwrap.wrap(text, width=84, break_long_words=False))
            f.write(f"{index}\n")
            f.write(
                f"{format_timestamp(item['start'], srt=True)} --> "
                f"{format_timestamp(item['end'], srt=True)}\n"
            )
            f.write(f"{wrapped}\n\n")


def save_vtt(path: Path, utterances: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for item in utterances:
            text = f"{item['speaker']}: {item['text']}"
            wrapped = "\n".join(textwrap.wrap(text, width=84, break_long_words=False))
            f.write(
                f"{format_timestamp(item['start'])} --> {format_timestamp(item['end'])}\n"
            )
            f.write(f"{wrapped}\n\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fully-local, resumable Whisper Turbo + pyannote Community-1 transcription "
            "with external crash-safe chunks and stable speaker IDs."
        )
    )

    parser.add_argument("audio", help="Input audio or video file")
    parser.add_argument("--output", default="output", help="Output/job directory")
    parser.add_argument(
        "--language",
        default=None,
        help="Whisper language code such as it/en/fr/de. Omit for auto-detection.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Whisper initial prompt for names, acronyms, products, technical terms, etc.",
    )
    parser.add_argument(
        "--chunk-minutes",
        type=float,
        default=10.0,
        help="Committed checkpoint size in minutes (default: 10)",
    )
    parser.add_argument(
        "--context-seconds",
        type=float,
        default=15.0,
        help="Extra audio read before and after each core chunk (default: 15)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help=(
            "Minimum speakers passed to pyannote PER CHUNK. Usually leave unset when "
            "chunking because some chunks may contain only one participant."
        ),
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum speakers passed to pyannote per chunk. Safe if it is a true global upper bound.",
    )
    parser.add_argument(
        "--single-speaker",
        action="store_true",
        help="Assume one speaker; skip pyannote and label every word SPEAKER_00.",
    )
    parser.add_argument(
        "--speaker-match-threshold",
        type=float,
        default=0.70,
        help=(
            "Cosine similarity used to recognize a returning speaker when overlap evidence "
            "is unavailable (default: 0.70; tune for your recordings)."
        ),
    )
    parser.add_argument(
        "--overlap-match-seconds",
        type=float,
        default=0.75,
        help="Shared-audio overlap needed to map the same local speaker across adjacent chunks.",
    )
    parser.add_argument(
        "--speaker-gap-tolerance",
        type=float,
        default=1.0,
        help="Maximum gap in seconds when assigning a Whisper word to the nearest speaker.",
    )
    parser.add_argument(
        "--utterance-max-seconds",
        type=float,
        default=15.0,
        help="Maximum output subtitle/utterance duration.",
    )
    parser.add_argument(
        "--utterance-gap-seconds",
        type=float,
        default=1.5,
        help="Split an utterance after this much silence.",
    )
    parser.add_argument(
        "--whisper-verbose",
        action="store_true",
        help="Print Whisper segment text while each chunk is decoded.",
    )
    parser.add_argument(
        "--force-whisper",
        action="store_true",
        help="Recompute every Whisper chunk even when checkpoints are valid.",
    )
    parser.add_argument(
        "--force-diarization",
        action="store_true",
        help="Recompute every pyannote chunk even when checkpoints are valid.",
    )
    parser.add_argument(
        "--rebuild-audio",
        action="store_true",
        help="Recreate the normalized 16 kHz mono WAV.",
    )

    args = parser.parse_args()

    if not math.isfinite(args.chunk_minutes) or args.chunk_minutes <= 0:
        parser.error("--chunk-minutes must be finite and > 0")
    if not math.isfinite(args.context_seconds) or args.context_seconds < 0:
        parser.error("--context-seconds must be finite and >= 0")
    if args.context_seconds >= args.chunk_minutes * 60:
        parser.error("--context-seconds must be smaller than the core chunk duration")
    if not math.isfinite(args.speaker_match_threshold) or not -1.0 <= args.speaker_match_threshold <= 1.0:
        parser.error("--speaker-match-threshold must be finite and between -1 and 1")
    if not math.isfinite(args.overlap_match_seconds) or args.overlap_match_seconds <= 0:
        parser.error("--overlap-match-seconds must be finite and > 0")
    if not math.isfinite(args.speaker_gap_tolerance) or args.speaker_gap_tolerance < 0:
        parser.error("--speaker-gap-tolerance must be finite and >= 0")
    if not math.isfinite(args.utterance_max_seconds) or args.utterance_max_seconds <= 0:
        parser.error("--utterance-max-seconds must be finite and > 0")
    if not math.isfinite(args.utterance_gap_seconds) or args.utterance_gap_seconds < 0:
        parser.error("--utterance-gap-seconds must be finite and >= 0")
    if args.min_speakers is not None and args.min_speakers < 1:
        parser.error("--min-speakers must be >= 1")
    if args.max_speakers is not None and args.max_speakers < 1:
        parser.error("--max-speakers must be >= 1")
    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        parser.error("--min-speakers cannot exceed --max-speakers")
    if args.single_speaker:
        if args.min_speakers not in (None, 1) or args.max_speakers not in (None, 1):
            parser.error("--single-speaker cannot be combined with speaker counts above 1")
        if args.force_diarization:
            parser.error("--force-diarization cannot be used with --single-speaker")

    return args


def main() -> None:
    args = parse_args()

    input_path = Path(args.audio).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with job_lock(output_dir):
            # The staging directory must be on the same volume as the exports.
            with tempfile.TemporaryDirectory(prefix=".voxtrail-exports-", dir=output_dir) as directory:
                run_transcription(args, input_path, output_dir, Path(directory))
    except JobBusyError as error:
        sys.exit(str(error))


def run_transcription(
    args: argparse.Namespace, input_path: Path, output_dir: Path, staged_dir: Path
) -> None:
    work_dir = output_dir / "work"
    whisper_checkpoint_dir = output_dir / "checkpoints" / "whisper"
    diarization_checkpoint_dir = output_dir / "checkpoints" / "diarization"

    validate_local_models(require_pyannote=not args.single_speaker)
    load_runtime_dependencies()

    print("=" * 80)
    print("LOCAL TRANSCRIPTION PIPELINE")
    print("=" * 80)
    print(f"Input:   {input_path}")
    print(f"Output:  {output_dir}")
    print(f"Whisper: {WHISPER_MODEL_PATH}")
    print("Speakers: one (pyannote skipped)" if args.single_speaker else f"pyannote:{PYANNOTE_MODEL_PATH}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:     {props.name}")
        print(f"VRAM:    {props.total_memory / (1024**3):.2f} GiB")
    else:
        print("GPU:     CUDA unavailable -> CPU fallback")

    state_path = output_dir / JOB_STATE_NAME
    save_json_atomic(state_path, {"status": "running"})
    normalized_wav = ensure_normalized_wav(
        input_path, work_dir, force=args.rebuild_audio
    )
    duration = wav_duration(normalized_wav)
    if duration <= 0:
        raise RuntimeError(f"Normalized audio is empty: {normalized_wav}")
    chunk_seconds = args.chunk_minutes * 60.0
    chunks = build_chunks(duration, chunk_seconds, args.context_seconds)

    audio_sha256 = file_sha256(normalized_wav)
    whisper_signature = whisper_stage_signature(
        normalized_wav, args.language, args.prompt, chunk_seconds, args.context_seconds,
        audio_sha256=audio_sha256,
    )
    if args.single_speaker:
        diarization_signature = "single-speaker-v1"
    else:
        diarization_signature = pyannote_stage_signature(
            normalized_wav, args.min_speakers, args.max_speakers, chunk_seconds, args.context_seconds,
            audio_sha256=audio_sha256,
        )
    whisper_generation, diarization_generation, force_manifest = prepare_force_run(
        output_dir,
        args.force_whisper,
        args.force_diarization,
        whisper_signature,
        diarization_signature,
    )

    print(f"Duration: {format_timestamp(duration)}")
    print(
        f"External checkpointing: {args.chunk_minutes:g} min core + "
        f"{args.context_seconds:g}s context on each side -> {len(chunks)} chunks"
    )

    # Stage 1: Whisper owns the GPU for the complete transcription pass.
    run_whisper_chunks(
        normalized_wav=normalized_wav,
        chunks=chunks,
        checkpoint_dir=whisper_checkpoint_dir,
        language=args.language,
        prompt=args.prompt,
        chunk_seconds=chunk_seconds,
        context_seconds=args.context_seconds,
        force_generation=whisper_generation,
        verbose=args.whisper_verbose,
        stage_signature=whisper_signature,
    )

    reconciliation_path = staged_dir / "speaker_reconciliation.json"
    if args.single_speaker:
        print("\nSTAGE 2 / 3 - SINGLE SPEAKER (PYANNOTE SKIPPED)")
        save_single_speaker_reconciliation(reconciliation_path)
        speaker_turns = []
    else:
        # Whisper is unloaded before pyannote takes over the GPU.
        run_pyannote_chunks(
            normalized_wav=normalized_wav,
            chunks=chunks,
            checkpoint_dir=diarization_checkpoint_dir,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            chunk_seconds=chunk_seconds,
            context_seconds=args.context_seconds,
            force_generation=diarization_generation,
            stage_signature=diarization_signature,
        )
        speaker_turns = reconcile_speakers(
            chunks=chunks,
            checkpoint_dir=diarization_checkpoint_dir,
            output_path=reconciliation_path,
            embedding_threshold=args.speaker_match_threshold,
            overlap_match_seconds=args.overlap_match_seconds,
        )

    words = load_all_words(chunks, whisper_checkpoint_dir)
    if args.single_speaker:
        words_with_speakers = [{**word, "speaker": "SPEAKER_00"} for word in words]
    else:
        words_with_speakers = assign_speakers_to_words(
            words,
            speaker_turns,
            gap_tolerance=args.speaker_gap_tolerance,
        )
    utterances = build_utterances(
        words_with_speakers,
        max_duration=args.utterance_max_seconds,
        max_gap=args.utterance_gap_seconds,
    )

    save_json_atomic(staged_dir / "words_with_speakers.json", words_with_speakers)
    save_json_atomic(staged_dir / "transcript.json", utterances)
    save_txt(staged_dir / "transcript.txt", utterances)
    save_srt(staged_dir / "transcript.srt", utterances)
    save_vtt(staged_dir / "transcript.vtt", utterances)
    from quality_report import write_quality_report
    write_quality_report(
        staged_dir,
        duration_seconds=duration,
        words=words_with_speakers,
        utterances=utterances,
        reconciliation=load_json(reconciliation_path),
        chunks={chunk["index"]: chunk for chunk in chunks},
    )
    publish_exports(staged_dir, output_dir)
    complete_force_run(force_manifest)
    save_json_atomic(state_path, {"status": "complete"})

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Words:      {len(words_with_speakers)}")
    print(f"Utterances: {len(utterances)}")
    print(f"TXT:        {output_dir / 'transcript.txt'}")
    print(f"JSON:       {output_dir / 'transcript.json'}")
    print(f"SRT:        {output_dir / 'transcript.srt'}")
    print(f"VTT:        {output_dir / 'transcript.vtt'}")
    print(f"Debug map:  {output_dir / reconciliation_path.name}")
    print(f"Quality:    {output_dir / 'quality_report.txt'}")
    print("\nIf the process stops, run the same command again. Valid completed chunks are reused.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear_gpu()
        print("\nInterrupted. Completed chunk checkpoints are safe; rerun the same command to resume.")
        sys.exit(130)
