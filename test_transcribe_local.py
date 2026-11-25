"""Focused regressions for the dependency-free parts of the transcription pipeline."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import quality_report
import setup_models
from job_state import EXPORT_NAMES, JOB_STATE_NAME, job_lock, publish_exports


SCRIPT = Path(__file__).with_name("transcribe_local.py")


def load_pipeline():
    """Import pure helpers without loading the large inference dependencies."""
    spec = importlib.util.spec_from_file_location("transcribe_local_review_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load_pipeline()


class PipelineRegressionTests(unittest.TestCase):
    def test_interrupted_normalization_cannot_reuse_audio_from_another_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [root / "first.wav", root / "second.wav"]
            for sample, path in enumerate(sources):
                with wave.open(str(path), "wb") as wav:
                    wav.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
                    wav.writeframes(sample.to_bytes(2, "little"))

            def normalize(command, *, check):
                whitelist = command[command.index("-protocol_whitelist") + 1]
                self.assertEqual(whitelist, "file,crypto,data")
                source = Path(command[command.index("-i") + 1])
                Path(command[-1]).write_bytes(source.read_bytes())

            save_json = pipeline.save_json_atomic

            def interrupt_publish(path, data):
                if data is not None:
                    raise KeyboardInterrupt
                save_json(path, data)

            with (
                patch.object(pipeline, "require_binary", return_value="ffmpeg"),
                patch.object(pipeline.subprocess, "run", side_effect=normalize),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                work = root / "work"
                wav = pipeline.ensure_normalized_wav(sources[0], work)
                with patch.object(pipeline, "save_json_atomic", side_effect=interrupt_publish):
                    with self.assertRaises(KeyboardInterrupt):
                        pipeline.ensure_normalized_wav(sources[1], work)
                self.assertEqual(wav.read_bytes(), sources[1].read_bytes())
                wav = pipeline.ensure_normalized_wav(sources[0], work)
                self.assertEqual(wav.read_bytes(), sources[0].read_bytes())

                # An older version may already have left mismatched WAV and metadata.
                meta_path = work / "audio_16k_mono.meta.json"
                legacy = pipeline.load_json(meta_path)
                del legacy["cache_version"]
                pipeline.save_json_atomic(meta_path, legacy)
                wav.write_bytes(sources[1].read_bytes())
                wav = pipeline.ensure_normalized_wav(sources[0], work)
                self.assertEqual(wav.read_bytes(), sources[0].read_bytes())

    def test_boundary_merge_removes_duplicate_core_words(self):
        chunks = pipeline.build_chunks(1200.0, 600.0, 15.0)
        previous = {"start": 599.7, "end": 600.1, "text": " hello", "probability": 0.9}
        current = {"start": 599.9, "end": 600.3, "text": " hello", "probability": 0.8}
        words = self.merge_words(chunks, [[previous], [current]])
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0], previous)

    def test_boundary_merge_recovers_word_outside_both_cores(self):
        chunks = pipeline.build_chunks(1200.0, 600.0, 15.0)
        previous = {"start": 599.9, "end": 600.3, "text": " hello", "probability": 0.9}
        current = {"start": 599.7, "end": 600.1, "text": " hello", "probability": 0.8}
        words = self.merge_words(chunks, [[previous], [current]])
        self.assertEqual(words, [previous])

    def test_boundary_merge_keeps_separate_repeated_words(self):
        chunks = pipeline.build_chunks(1200.0, 600.0, 15.0)
        previous = {"start": 599.90, "end": 599.98, "text": " no", "probability": 0.9}
        current = {"start": 600.00, "end": 600.08, "text": " no", "probability": 0.8}
        words = self.merge_words(chunks, [[previous], [current]])
        self.assertEqual(words, [previous, current])

    def merge_words(self, chunks, words_by_chunk):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            for chunk, words in zip(chunks, words_by_chunk):
                pipeline.save_json_atomic(
                    checkpoint_dir / f"chunk_{chunk['index']:06d}.json",
                    {"words_extended": words},
                )
            with contextlib.redirect_stdout(io.StringIO()):
                return pipeline.load_all_words(chunks, checkpoint_dir)

    def test_matching_uses_best_total_overlap(self):
        candidates = [
            (5.0, "local_a", "global_a"),
            (4.9, "local_a", "global_b"),
            (4.8, "local_b", "global_a"),
            (0.1, "local_b", "global_b"),
        ]
        self.assertEqual(
            {(local, global_id) for _, local, global_id in pipeline.maximum_weight_pairs(candidates)},
            {("local_a", "global_b"), ("local_b", "global_a")},
        )

    def test_matching_score_matches_brute_force_on_small_cases(self):
        rng = random.Random(17)
        for _ in range(40):
            scores = {
                (local, global_id): rng.randint(1, 10)
                for local in range(3)
                for global_id in range(3)
                if rng.random() < 0.7
            }
            candidates = [
                (score, str(local), str(global_id))
                for (local, global_id), score in scores.items()
            ]
            matched = pipeline.maximum_weight_pairs(candidates)
            actual = sum(score for score, _, _ in matched)

            def best_from(local, used):
                if local == 3:
                    return 0
                best = best_from(local + 1, used)
                for global_id in range(3):
                    if global_id not in used and (local, global_id) in scores:
                        best = max(
                            best,
                            scores[(local, global_id)] + best_from(local + 1, used | {global_id}),
                        )
                return best

            self.assertEqual(actual, best_from(0, set()))

    def test_malformed_checkpoint_is_rejected(self):
        chunk = pipeline.build_chunks(10.0, 600.0, 0.0)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk.json"
            pipeline.save_json_atomic(
                path,
                {"stage": "whisper", "stage_signature": "sig", "chunk": chunk},
            )
            self.assertIsNone(pipeline.load_valid_chunk_checkpoint(path, "sig", chunk, "whisper"))
            pipeline.save_json_atomic(
                path,
                {
                    "stage": "whisper",
                    "stage_signature": "sig",
                    "chunk": chunk,
                    "words_extended": [],
                    "force_generation": "run-1",
                },
            )
            self.assertIsNotNone(pipeline.load_valid_chunk_checkpoint(path, "sig", chunk, "whisper"))
            self.assertIsNotNone(
                pipeline.load_valid_chunk_checkpoint(path, "sig", chunk, "whisper", "run-1")
            )
            self.assertIsNone(
                pipeline.load_valid_chunk_checkpoint(path, "sig", chunk, "whisper", "run-2")
            )

    def test_timed_checkpoint_fields_reject_numeric_strings_and_invalid_probabilities(self):
        chunk = pipeline.build_chunks(10.0, 600.0, 0.0)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk.json"
            for field, value in (("start", "1.0"), ("start", True), ("end", float("inf")),
                                 ("probability", "0.9"), ("probability", 1.5)):
                with self.subTest(field=field, value=value):
                    word = {"start": 1.0, "end": 2.0, "text": " hi", "probability": 0.9}
                    word[field] = value
                    pipeline.save_json_atomic(path, {
                        "stage": "whisper", "stage_signature": "sig", "chunk": chunk,
                        "words_extended": [word],
                    })
                    cached = pipeline.load_valid_chunk_checkpoint(path, "sig", chunk, "whisper")
                    self.assertIsNone(cached)

    def test_invalid_force_manifest_is_ignored_and_can_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "work" / "force_run.json"
            for manifest in ({"complete": False},
                             {"complete": False, "generation": "run-1", "requested": []}):
                with self.subTest(manifest=manifest):
                    pipeline.save_json_atomic(path, manifest)
                    self.assertEqual(
                        pipeline.prepare_force_run(output, False, False, "w", "d"),
                        (None, None, None),
                    )
            whisper, diarization, manifest_path = pipeline.prepare_force_run(output, True, False, "w", "d")
            self.assertTrue(whisper)
            self.assertIsNone(diarization)
            self.assertEqual(manifest_path, path)

    def test_force_run_resumes_then_starts_new_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = pipeline.prepare_force_run(output, True, False, "whisper-sig", "diarization-sig")
            resumed = pipeline.prepare_force_run(output, True, False, "whisper-sig", "diarization-sig")
            self.assertEqual(first, resumed)
            self.assertIsNotNone(first[0])
            self.assertIsNone(first[1])
            pipeline.complete_force_run(first[2])
            new_run = pipeline.prepare_force_run(output, True, False, "whisper-sig", "diarization-sig")
            self.assertNotEqual(first[0], new_run[0])

    def test_local_whisper_load_applies_turbo_alignment_heads_on_device(self):
        upstream = Mock()
        upstream._ALIGNMENT_HEADS = {"turbo": b"turbo-mask"}
        model = upstream.load_model.return_value
        aligned_on_device = model.alignment_heads.to.return_value
        with patch.object(pipeline, "whisper", upstream):
            self.assertIs(pipeline.load_local_whisper("cuda"), model)
        upstream.load_model.assert_called_once_with(str(pipeline.WHISPER_MODEL_PATH), device="cuda")
        model.set_alignment_heads.assert_called_once_with(b"turbo-mask")
        self.assertIs(model.alignment_heads, aligned_on_device)

    def test_interrupted_forced_whisper_pass_skips_completed_chunk(self):
        chunk = pipeline.build_chunks(10.0, 600.0, 0.0)[0]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            pipeline.save_json_atomic(
                checkpoint_dir / "chunk_000000.json",
                {
                    "stage": "whisper",
                    "stage_signature": "sig",
                    "force_generation": "run-1",
                    "chunk": chunk,
                    "words_extended": [],
                },
            )
            with (
                patch.object(pipeline, "whisper_stage_signature", return_value="sig"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                pipeline.run_whisper_chunks(
                    normalized_wav=Path("unused.wav"),
                    chunks=[chunk],
                    checkpoint_dir=checkpoint_dir,
                    language=None,
                    prompt=None,
                    chunk_seconds=600.0,
                    context_seconds=0.0,
                    force_generation="run-1",
                    verbose=False,
                )

    def test_offline_flags_override_inherited_online_settings(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import os, transcribe_local; print(os.environ['HF_HUB_OFFLINE'], "
             "os.environ['TRANSFORMERS_OFFLINE'], os.environ['PYANNOTE_METRICS_ENABLED'])"],
            cwd=SCRIPT.parent,
            env={**os.environ, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0",
                 "PYANNOTE_METRICS_ENABLED": "1"},
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(result.stdout.strip(), "1 1 0")

    def test_help_exits_without_loading_inference_dependencies(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--chunk-minutes", result.stdout)
        self.assertIsNone(pipeline.torch)
        self.assertIsNone(pipeline.whisper)

    def test_utterance_joins_missing_spaces_and_preserves_end(self):
        words = [
            {"speaker": "S", "start": 0.0, "end": 1.0, "text": "Hello"},
            {"speaker": "S", "start": 0.5, "end": 0.8, "text": "world"},
        ]
        utterances = pipeline.build_utterances(words, 15.0, 1.5)
        self.assertEqual(utterances[0]["text"], "Hello world")
        self.assertEqual(utterances[0]["end"], 1.0)

    def test_utterance_spacing_respects_punctuation_and_cjk(self):
        self.assertTrue(pipeline.needs_word_separator("Hello,", "world"))
        self.assertFalse(pipeline.needs_word_separator("你", "好"))
        self.assertFalse(pipeline.needs_word_separator("don", "’t"))
        self.assertFalse(pipeline.needs_word_separator("ready", "-to"))
        self.assertTrue(pipeline.needs_word_separator("ready", "-"))
        self.assertTrue(pipeline.needs_word_separator("'Hello'", "world"))

    def test_invalid_cli_values_are_rejected(self):
        invalid = [
            ("--overlap-match-seconds", "0"),
            ("--speaker-gap-tolerance", "-1"),
            ("--utterance-max-seconds", "-2"),
            ("--chunk-minutes", "nan"),
        ]
        for option, value in invalid:
            with self.subTest(option=option), patch.object(sys, "argv", ["script", "audio.wav", option, value]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    pipeline.parse_args()
                self.assertEqual(error.exception.code, 2)

    def test_single_speaker_cli_rejects_conflicting_options(self):
        with patch.object(sys, "argv", ["script", "audio.wav", "--single-speaker", "--min-speakers", "1"]):
            self.assertTrue(pipeline.parse_args().single_speaker)
        for option in (("--max-speakers", "2"), ("--force-diarization",)):
            with self.subTest(option=option), patch.object(
                sys, "argv", ["script", "audio.wav", "--single-speaker", *option]
            ):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    pipeline.parse_args()
                self.assertEqual(error.exception.code, 2)

    def test_single_speaker_model_check_needs_only_whisper(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "large-v3-turbo.pt"
            model.write_bytes(b"model")
            with (
                patch.object(pipeline, "WHISPER_MODEL_PATH", model),
                patch.object(pipeline, "PYANNOTE_MODEL_PATH", Path(directory) / "missing-pyannote"),
            ):
                pipeline.validate_local_models(require_pyannote=False)
                with self.assertRaisesRegex(RuntimeError, "pyannote Community-1 model missing"):
                    pipeline.validate_local_models(require_pyannote=True)

    def test_single_speaker_run_skips_pyannote_and_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            audio.write_bytes(b"test")
            output = root / "result"
            words = [
                {"start": 0.2, "end": 0.5, "text": "Hello", "probability": 0.9},
                {"start": 0.6, "end": 1.0, "text": " world", "probability": 0.8},
            ]
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            with (
                patch.object(sys, "argv", ["script", str(audio), "--single-speaker", "--output", str(output)]),
                patch.object(pipeline, "validate_local_models") as validate,
                patch.object(pipeline, "load_runtime_dependencies"),
                patch.object(pipeline, "torch", fake_torch),
                patch.object(pipeline, "ensure_normalized_wav", return_value=audio),
                patch.object(pipeline, "wav_duration", return_value=2.0),
                patch.object(pipeline, "whisper_stage_signature", return_value="whisper-test"),
                patch.object(pipeline, "pyannote_stage_signature", side_effect=AssertionError("pyannote signature used")),
                patch.object(pipeline, "run_whisper_chunks"),
                patch.object(pipeline, "run_pyannote_chunks", side_effect=AssertionError("pyannote used")),
                patch.object(pipeline, "reconcile_speakers", side_effect=AssertionError("diarization used")),
                patch.object(pipeline, "load_all_words", return_value=words),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                pipeline.main()
            validate.assert_called_once_with(require_pyannote=False)
            state = pipeline.load_json(output / JOB_STATE_NAME)
            self.assertEqual(state["status"], "complete")
            assigned = json.loads((output / "words_with_speakers.json").read_text(encoding="utf-8"))
            self.assertEqual({word["speaker"] for word in assigned}, {"SPEAKER_00"})
            self.assertEqual(json.loads((output / "speaker_reconciliation.json").read_text(encoding="utf-8"))["mode"], "single_speaker")
            self.assertEqual(json.loads((output / "quality_report.json").read_text(encoding="utf-8"))["summary"]["word_count"], 2)

    def test_incomplete_job_and_active_writer_are_rejected_by_report_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            command = [sys.executable, str(SCRIPT.with_name("quality_report.py")), str(output)]
            with job_lock(output):
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                self.assertIn("already in use", result.stderr)
            pipeline.save_json_atomic(output / JOB_STATE_NAME, {"status": "running"})
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("Job is incomplete", result.stderr)

    def test_interrupted_export_publication_cannot_leave_mixed_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            staged = output / "staged"
            staged.mkdir()
            for name in EXPORT_NAMES:
                (output / name).write_text("old", encoding="utf-8")
                (staged / name).write_text("new", encoding="utf-8")

            replace = Path.replace

            def interrupt_second_export(source, destination):
                if source.name == EXPORT_NAMES[1]:
                    raise KeyboardInterrupt
                replace(source, destination)

            with patch.object(Path, "replace", interrupt_second_export):
                with self.assertRaises(KeyboardInterrupt):
                    publish_exports(staged, output)
            self.assertEqual((output / EXPORT_NAMES[0]).read_text(encoding="utf-8"), "new")
            self.assertFalse(any((output / name).exists() for name in EXPORT_NAMES[1:]))

    def test_quality_report_flags_review_passages_without_models(self):
        words = [
            {"start": 0.0, "end": 0.3, "text": "Hello", "probability": 0.9, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 1.2, "text": "unclear", "probability": 0.05, "speaker": "UNKNOWN"},
            {"start": 1.3, "end": 1.5, "text": "words", "probability": 0.2, "speaker": "UNKNOWN"},
            {"start": 11.0, "end": 11.2, "text": "Brief", "probability": 0.8, "speaker": "SPEAKER_01"},
        ]
        utterances = [{"start": 11.0, "end": 11.2, "text": "Brief", "speaker": "SPEAKER_01"}]
        reconciliation = {
            "global_speakers": ["SPEAKER_00", "SPEAKER_01"],
            "embedding_match_threshold": 0.7,
            "chunks": [{"chunk_index": 1, "mapping": {"local": "SPEAKER_01"},
                        "evidence": {"local": {"method": "embedding", "cosine_similarity": 0.72}}}],
        }
        report = quality_report.build_quality_report(
            words, utterances, reconciliation, 20.0,
            {1: {"core_start": 10.0, "core_end": 20.0}},
        )
        self.assertEqual(report["summary"]["unknown_word_count"], 2)
        self.assertEqual(report["review"]["unknown_speaker"][0]["words"], 2)
        self.assertIn("SPEAKER_01", {item["speaker"] for item in report["review"]["short_speaker"]})
        self.assertEqual(report["review"]["low_word_score"][0]["low_score_words"], 2)
        self.assertEqual(report["review"]["weak_speaker_match"][0]["start"], 11.0)

    def test_quality_report_cli_uses_existing_job_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            pipeline.save_json_atomic(output / "words_with_speakers.json", [
                {"start": 1.0, "end": 1.5, "text": " hello", "probability": 0.8, "speaker": "SPEAKER_00"},
            ])
            pipeline.save_json_atomic(output / "transcript.json", [
                {"start": 1.0, "end": 1.5, "text": "hello", "speaker": "SPEAKER_00"},
            ])
            pipeline.save_json_atomic(output / "speaker_reconciliation.json", {
                "mode": "single_speaker", "global_speakers": ["SPEAKER_00"], "chunks": [],
            })
            pipeline.save_json_atomic(output / "checkpoints" / "whisper" / "chunk_000000.json", {
                "chunk": {"index": 0, "core_start": 0.0, "core_end": 10.0},
            })
            result = subprocess.run(
                [sys.executable, str(SCRIPT.with_name("quality_report.py")), str(output)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["duration_seconds"], 10.0)
            self.assertEqual(report["review"]["short_speaker"], [])

    def test_whisper_only_setup_skips_hugging_face_sign_in(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "large-v3-turbo.pt"

            def download(url, destination, _in_memory):
                self.assertEqual(Path(destination), model.parent)
                model.write_bytes(b"model")

            fake_whisper = SimpleNamespace(
                _MODELS={"turbo": "https://example.invalid/large-v3-turbo.pt"},
                _download=download,
            )
            with (
                patch.object(setup_models, "WHISPER_PATH", model),
                patch.object(setup_models, "require_login", side_effect=AssertionError("login used")),
                patch.object(setup_models, "show_linux_build_tools"),
                patch.dict(sys.modules, {"whisper": fake_whisper}),
                patch.dict("os.environ", {"HF_HUB_OFFLINE": "0"}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                setup_models.download_models(whisper_only=True)
            self.assertTrue(model.is_file())


if __name__ == "__main__":
    unittest.main()
