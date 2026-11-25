"""Local Whisper Turbo loading and chunk-level transcription."""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any

from .. import audio as audio_io
from .. import checkpoints, local_models, runtime
from ..transcript import format_timestamp


def load_local_whisper(device: str) -> Any:
    model = runtime.whisper.load_model(str(local_models.WHISPER_MODEL_PATH), device=device)
    # Upstream applies these heads for the model name, but skips them for a path.
    model.set_alignment_heads(runtime.whisper._ALIGNMENT_HEADS["turbo"])
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
    signature = stage_signature or checkpoints.whisper_stage_signature(
        normalized_wav, language, prompt, chunk_seconds, context_seconds
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for chunk in chunks:
        path = checkpoint_dir / f"chunk_{chunk['index']:06d}.json"
        if checkpoints.load_valid_chunk_checkpoint(path, signature, chunk, "whisper", force_generation) is None:
            pending.append(chunk)

    print("\n" + "=" * 80)
    print("STAGE 1 / 3 - WHISPER TURBO")
    print("=" * 80)
    print(f"Chunks total: {len(chunks)} | pending: {len(pending)}")

    if not pending:
        print("All Whisper chunks already checkpointed. Skipping model load.")
        return

    device = "cuda" if runtime.torch.cuda.is_available() else "cpu"
    print(f"Loading local Whisper model: {local_models.WHISPER_MODEL_PATH}")
    print(f"Device: {device}")

    model = load_local_whisper(device)
    runtime.print_gpu_memory("After Whisper load: ")

    try:
        for n, chunk in enumerate(chunks, start=1):
            checkpoint = checkpoint_dir / f"chunk_{chunk['index']:06d}.json"

            if checkpoints.load_valid_chunk_checkpoint(checkpoint, signature, chunk, "whisper", force_generation):
                print(f"[Whisper {n}/{len(chunks)}] checkpoint exists -> skip")
                continue

            print(
                f"\n[Whisper {n}/{len(chunks)}] "
                f"core {format_timestamp(chunk['core_start'])} -> {format_timestamp(chunk['core_end'])} | "
                f"read {format_timestamp(chunk['ext_start'])} -> {format_timestamp(chunk['ext_end'])}"
            )

            audio = audio_io.read_wav_interval(normalized_wav, chunk["ext_start"], chunk["ext_end"])

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
            checkpoints.save_json_atomic(checkpoint, checkpoint_data)
            print(f"Saved {len(words_extended)} extended words -> {checkpoint}")

            del result
            del audio
            gc.collect()

    finally:
        del model
        runtime.clear_gpu()
        runtime.print_gpu_memory("After Whisper unload: ")
