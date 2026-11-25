"""Local Community-1 loading and chunk-level speaker diarization."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from .. import audio as audio_io
from .. import checkpoints, local_models, runtime
from ..transcript import format_timestamp


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
    signature = stage_signature or checkpoints.pyannote_stage_signature(
        normalized_wav, min_speakers, max_speakers, chunk_seconds, context_seconds
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for chunk in chunks:
        path = checkpoint_dir / f"chunk_{chunk['index']:06d}.json"
        if checkpoints.load_valid_chunk_checkpoint(path, signature, chunk, "diarization", force_generation) is None:
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

    device = runtime.torch.device("cuda" if runtime.torch.cuda.is_available() else "cpu")
    print(f"Loading local pyannote model: {local_models.PYANNOTE_MODEL_PATH}")
    print(f"Device: {device}")

    pipeline = Pipeline.from_pretrained(str(local_models.PYANNOTE_MODEL_PATH))
    pipeline.to(device)
    runtime.print_gpu_memory("After pyannote load: ")

    try:
        for n, chunk in enumerate(chunks, start=1):
            checkpoint = checkpoint_dir / f"chunk_{chunk['index']:06d}.json"

            if checkpoints.load_valid_chunk_checkpoint(checkpoint, signature, chunk, "diarization", force_generation):
                print(f"[pyannote {n}/{len(chunks)}] checkpoint exists -> skip")
                continue

            print(
                f"\n[pyannote {n}/{len(chunks)}] "
                f"core {format_timestamp(chunk['core_start'])} -> {format_timestamp(chunk['core_end'])} | "
                f"read {format_timestamp(chunk['ext_start'])} -> {format_timestamp(chunk['ext_end'])}"
            )

            audio = audio_io.read_wav_interval(normalized_wav, chunk["ext_start"], chunk["ext_end"])
            waveform = runtime.torch.from_numpy(audio).unsqueeze(0)

            kwargs: dict[str, Any] = {}
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers

            with ProgressHook() as hook:
                output = pipeline(
                    {"waveform": waveform, "sample_rate": audio_io.SAMPLE_RATE},
                    hook=hook,
                    **kwargs,
                )

            regular = output.speaker_diarization
            exclusive = output.exclusive_speaker_diarization
            labels = [str(label) for label in regular.labels()]
            embeddings_array = runtime.np.asarray(output.speaker_embeddings, dtype=runtime.np.float32)

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
            checkpoints.save_json_atomic(checkpoint, checkpoint_data)
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
        runtime.clear_gpu()
        runtime.print_gpu_memory("After pyannote unload: ")
