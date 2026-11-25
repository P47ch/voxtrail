"""Checkpoint serialization, stage fingerprints, and forced-run state."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any

from . import local_models, runtime
from .job_state import atomic_text_file


CHECKPOINT_SCHEMA_VERSION = 2


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if runtime.np is not None and isinstance(value, runtime.np.ndarray):
        return value.tolist()
    if runtime.np is not None and isinstance(value, runtime.np.generic):
        return value.item()
    if runtime.torch is not None and runtime.torch.is_tensor(value):
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
        "normalized_audio_sha256": audio_sha256 or local_models.file_sha256(normalized_wav),
        "model": file_identity(local_models.WHISPER_MODEL_PATH),
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
        "normalized_audio_sha256": audio_sha256 or local_models.file_sha256(normalized_wav),
        "model": directory_identity(local_models.PYANNOTE_MODEL_PATH),
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "chunk_seconds": chunk_seconds,
        "context_seconds": context_seconds,
        "exclusive_diarization": True,
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
