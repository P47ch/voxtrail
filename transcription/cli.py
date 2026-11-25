"""Command-line arguments and orchestration of a transcription job."""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

from . import audio, checkpoints, local_models, runtime, speakers, transcript
from .inference import diarization, whisper
from .job_state import JOB_STATE_NAME, JobBusyError, job_lock, publish_exports


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

    local_models.validate_local_models(require_pyannote=not args.single_speaker)
    runtime.load_runtime_dependencies()

    print("=" * 80)
    print("LOCAL TRANSCRIPTION PIPELINE")
    print("=" * 80)
    print(f"Input:   {input_path}")
    print(f"Output:  {output_dir}")
    print(f"Whisper: {local_models.WHISPER_MODEL_PATH}")
    print("Speakers: one (pyannote skipped)" if args.single_speaker else f"pyannote:{local_models.PYANNOTE_MODEL_PATH}")

    if runtime.torch.cuda.is_available():
        props = runtime.torch.cuda.get_device_properties(0)
        print(f"GPU:     {props.name}")
        print(f"VRAM:    {props.total_memory / (1024**3):.2f} GiB")
    else:
        print("GPU:     CUDA unavailable -> CPU fallback")

    state_path = output_dir / JOB_STATE_NAME
    checkpoints.save_json_atomic(state_path, {"status": "running"})
    normalized_wav = audio.ensure_normalized_wav(
        input_path, work_dir, force=args.rebuild_audio
    )
    duration = audio.wav_duration(normalized_wav)
    if duration <= 0:
        raise RuntimeError(f"Normalized audio is empty: {normalized_wav}")
    chunk_seconds = args.chunk_minutes * 60.0
    chunks = audio.build_chunks(duration, chunk_seconds, args.context_seconds)

    audio_sha256 = local_models.file_sha256(normalized_wav)
    whisper_signature = checkpoints.whisper_stage_signature(
        normalized_wav, args.language, args.prompt, chunk_seconds, args.context_seconds,
        audio_sha256=audio_sha256,
    )
    if args.single_speaker:
        diarization_signature = "single-speaker-v1"
    else:
        diarization_signature = checkpoints.pyannote_stage_signature(
            normalized_wav, args.min_speakers, args.max_speakers, chunk_seconds, args.context_seconds,
            audio_sha256=audio_sha256,
        )
    whisper_generation, diarization_generation, force_manifest = checkpoints.prepare_force_run(
        output_dir,
        args.force_whisper,
        args.force_diarization,
        whisper_signature,
        diarization_signature,
    )

    print(f"Duration: {transcript.format_timestamp(duration)}")
    print(
        f"External checkpointing: {args.chunk_minutes:g} min core + "
        f"{args.context_seconds:g}s context on each side -> {len(chunks)} chunks"
    )

    # Stage 1: Whisper owns the GPU for the complete transcription pass.
    whisper.run_whisper_chunks(
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
        speakers.save_single_speaker_reconciliation(reconciliation_path)
        speaker_turns = []
    else:
        # Whisper is unloaded before pyannote takes over the GPU.
        diarization.run_pyannote_chunks(
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
        speaker_turns = speakers.reconcile_speakers(
            chunks=chunks,
            checkpoint_dir=diarization_checkpoint_dir,
            output_path=reconciliation_path,
            embedding_threshold=args.speaker_match_threshold,
            overlap_match_seconds=args.overlap_match_seconds,
        )

    words = transcript.load_all_words(chunks, whisper_checkpoint_dir)
    if args.single_speaker:
        words_with_speakers = [{**word, "speaker": "SPEAKER_00"} for word in words]
    else:
        words_with_speakers = speakers.assign_speakers_to_words(
            words,
            speaker_turns,
            gap_tolerance=args.speaker_gap_tolerance,
        )
    utterances = transcript.build_utterances(
        words_with_speakers,
        max_duration=args.utterance_max_seconds,
        max_gap=args.utterance_gap_seconds,
    )

    checkpoints.save_json_atomic(staged_dir / "words_with_speakers.json", words_with_speakers)
    checkpoints.save_json_atomic(staged_dir / "transcript.json", utterances)
    transcript.save_txt(staged_dir / "transcript.txt", utterances)
    transcript.save_srt(staged_dir / "transcript.srt", utterances)
    transcript.save_vtt(staged_dir / "transcript.vtt", utterances)
    from .quality_report import write_quality_report
    write_quality_report(
        staged_dir,
        duration_seconds=duration,
        words=words_with_speakers,
        utterances=utterances,
        reconciliation=checkpoints.load_json(reconciliation_path),
        chunks={chunk["index"]: chunk for chunk in chunks},
    )
    publish_exports(staged_dir, output_dir)
    checkpoints.complete_force_run(force_manifest)
    checkpoints.save_json_atomic(state_path, {
        "status": "complete",
        "duration_seconds": duration,
        "chunks": chunks,
    })

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
