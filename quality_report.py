"""Generate review cues from a completed transcription without loading any models."""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from functools import partial
from itertools import accumulate
from pathlib import Path
from typing import Any

from job_state import JOB_STATE_NAME, JobBusyError, atomic_text_file, job_lock


LOW_WORD_SCORE = 0.30
VERY_LOW_WORD_SCORE = 0.10
SHORT_SPEAKER_WORDS = 20
WEAK_EMBEDDING_MARGIN = 0.05
TEXT_ITEMS_PER_SECTION = 20


def timestamp(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, remainder = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{remainder:03d}"


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed-job file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_text_atomic(path: Path, content: str) -> None:
    with atomic_text_file(path, newline="\n") as file:
        file.write(content)


def _checkpoint_chunks(output_dir: Path) -> dict[int, dict[str, Any]]:
    # Legacy jobs have no completed-job metadata. Chunk zero identifies the
    # latest stage run, so ignore trailing files from an older stage signature.
    cached: dict[int, dict[str, Any]] = {}
    for path in (output_dir / "checkpoints" / "whisper").glob("chunk_*.json"):
        try:
            data = _read_json(path)
            cached[int(data["chunk"]["index"])] = data
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if 0 not in cached:
        return {}

    signature = cached[0].get("stage_signature")
    chunks: dict[int, dict[str, Any]] = {}
    for index in range(len(cached)):
        data = cached.get(index)
        if data is None or data.get("stage_signature") != signature:
            break
        chunks[index] = data["chunk"]
    return chunks


def _excerpt(
    words: list[dict[str, Any]],
    starts: list[float],
    prefix_ends: list[float],
    start: float,
    end: float,
) -> str:
    first = bisect_left(prefix_ends, start - 2)
    stop = bisect_right(starts, end + 2, lo=first)
    parts = []
    length = 0
    for index in range(first, stop):
        word = words[index]
        if float(word["end"]) < start - 2:
            continue
        text = str(word.get("text", "")).strip()
        if text:
            length += len(text) + bool(parts)
            parts.append(text)
            if length > 160:
                return " ".join(parts)[:157] + "..."
    return " ".join(parts)


def _group_words(
    words: list[dict[str, Any]], *, max_gap: float = 2.0, max_span: float | None = None
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for word in sorted(words, key=lambda item: (item["start"], item["end"])):
        if (
            groups
            and float(word["start"]) - float(groups[-1][-1]["end"]) <= max_gap
            and (max_span is None or float(word["end"]) - float(groups[-1][0]["start"]) <= max_span)
        ):
            groups[-1].append(word)
        else:
            groups.append([word])
    return groups


def build_quality_report(
    words: list[dict[str, Any]],
    utterances: list[dict[str, Any]],
    reconciliation: dict[str, Any],
    duration_seconds: float,
    chunks: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return review hints, not an estimated transcription accuracy score."""
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        raise ValueError("Recording duration must be finite and nonnegative")
    chunks = chunks or {}
    words = sorted(words, key=lambda item: (item["start"], item["end"]))
    starts = [float(word["start"]) for word in words]
    # Prefix maxima let an earlier, overlapping word survive the lower bound.
    prefix_ends = list(accumulate((float(word["end"]) for word in words), max))
    excerpt = partial(_excerpt, words, starts, prefix_ends)
    mode = reconciliation.get("mode", "diarization")

    speaker_words: dict[str, list[dict[str, Any]]] = defaultdict(list)
    speaker_utterances: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for word in words:
        speaker_words[str(word["speaker"])].append(word)
    for utterance in utterances:
        speaker_utterances[str(utterance["speaker"])].append(utterance)

    known_speakers = sorted(
        (set(reconciliation.get("global_speakers", [])) | set(speaker_words)) - {"UNKNOWN"}
    )
    speakers = []
    for speaker in known_speakers:
        own_words = speaker_words[speaker]
        speakers.append({
            "speaker": speaker,
            "words": len(own_words),
            "utterances": len(speaker_utterances[speaker]),
            "first_start": min((float(word["start"]) for word in own_words), default=None),
            "last_end": max((float(word["end"]) for word in own_words), default=None),
        })

    unknown_words = speaker_words["UNKNOWN"]
    unknown_groups = [{
        "start": float(group[0]["start"]),
        "end": max(float(word["end"]) for word in group),
        "words": len(group),
        "excerpt": excerpt(float(group[0]["start"]), float(group[-1]["end"])),
    } for group in _group_words(unknown_words)]

    scored_words = [
        word for word in words
        if isinstance(word.get("probability"), (int, float))
        and math.isfinite(float(word["probability"]))
    ]
    low_words = [word for word in scored_words if float(word["probability"]) < LOW_WORD_SCORE]
    low_score_groups = []
    for group in _group_words(low_words, max_span=15.0):
        minimum = min(float(word["probability"]) for word in group)
        if len(group) < 2 and minimum >= VERY_LOW_WORD_SCORE:
            continue
        start = float(group[0]["start"])
        end = max(float(word["end"]) for word in group)
        low_score_groups.append({
            "start": start,
            "end": end,
            "low_score_words": len(group),
            "minimum_word_score": round(minimum, 3),
            "excerpt": excerpt(start, end),
        })
    low_score_groups.sort(key=lambda item: (item["minimum_word_score"], -item["low_score_words"], item["start"]))

    short_speakers = []
    if mode != "single_speaker" and len(known_speakers) > 1:
        for speaker in speakers:
            if speaker["words"] >= SHORT_SPEAKER_WORDS:
                continue
            appearances = [
                {
                    "start": float(item["start"]),
                    "end": float(item["end"]),
                    "excerpt": str(item["text"])[:160],
                }
                for item in speaker_utterances[speaker["speaker"]][:10]
            ]
            short_speakers.append({**speaker, "appearances": appearances})
    short_speakers.sort(key=lambda item: (item["words"], item["speaker"]))

    weak_matches = []
    threshold = reconciliation.get("embedding_match_threshold")
    if isinstance(threshold, (int, float)) and math.isfinite(float(threshold)):
        for chunk_report in reconciliation.get("chunks", []):
            index = int(chunk_report["chunk_index"])
            chunk = chunks.get(index, {})
            start = float(chunk.get("core_start", 0.0)) if chunk else None
            end = float(chunk.get("core_end", 0.0)) if chunk else None
            for local, evidence in chunk_report.get("evidence", {}).items():
                if evidence.get("method") != "embedding":
                    continue
                similarity = float(evidence["cosine_similarity"])
                if similarity > float(threshold) + WEAK_EMBEDDING_MARGIN:
                    continue
                speaker = chunk_report.get("mapping", {}).get(local, "UNKNOWN")
                review_start = start
                if start is not None and end is not None:
                    spoken = speaker_words[speaker]
                    first = bisect_left(spoken, start, key=lambda word: float(word["start"]))
                    if first < len(spoken) and float(spoken[first]["start"]) < end:
                        review_start = float(spoken[first]["start"])
                weak_matches.append({
                    "chunk_index": index,
                    "speaker": speaker,
                    "start": review_start,
                    "cosine_similarity": round(similarity, 3),
                    "threshold": float(threshold),
                    "excerpt": excerpt(review_start, review_start) if review_start is not None else "",
                })
    weak_matches.sort(key=lambda item: (item["cosine_similarity"], item["chunk_index"]))

    return {
        "schema_version": 1,
        "mode": mode,
        "note": "Review cues are heuristics. Word scores are not an accuracy percentage; listen before correcting.",
        "review_rules": {
            "low_word_score_below": LOW_WORD_SCORE,
            "short_speaker_fewer_than_words": SHORT_SPEAKER_WORDS,
            "weak_embedding_margin_above_threshold": WEAK_EMBEDDING_MARGIN,
        },
        "summary": {
            "duration_seconds": duration_seconds,
            "word_count": len(words),
            "utterance_count": len(utterances),
            "speaker_count": len(known_speakers),
            "unknown_word_count": len(unknown_words),
            "scored_word_count": len(scored_words),
            "low_score_word_count": len(low_words),
        },
        "speakers": speakers,
        "review": {
            "unknown_speaker": unknown_groups,
            "short_speaker": short_speakers,
            "low_word_score": low_score_groups,
            "weak_speaker_match": weak_matches,
        },
    }


def render_quality_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "QUALITY REVIEW REPORT",
        "=====================",
        f"Mode: {report['mode']}",
        f"Duration: {timestamp(summary['duration_seconds'])}",
        f"Words: {summary['word_count']} | Utterances: {summary['utterance_count']}",
        f"Speakers: {summary['speaker_count']} | UNKNOWN words: {summary['unknown_word_count']}",
        f"Words with score below {LOW_WORD_SCORE:.2f}: {summary['low_score_word_count']} / {summary['scored_word_count']}",
        "",
        report["note"],
        "",
        "SPEAKERS",
        "--------",
    ]
    for speaker in report["speakers"]:
        first = timestamp(speaker["first_start"]) if speaker["first_start"] is not None else "no words"
        lines.append(f"{speaker['speaker']}: {speaker['words']} words, {speaker['utterances']} utterances; first at {first}")
    if not report["speakers"]:
        lines.append("No speakers detected.")

    sections = [
        ("UNKNOWN SPEAKER PASSAGES", "unknown_speaker"),
        ("SHORT SPEAKER APPEARANCES", "short_speaker"),
        ("LOW WORD SCORE PASSAGES", "low_word_score"),
        ("SPEAKER MATCHES NEAR THE THRESHOLD", "weak_speaker_match"),
    ]
    for title, key in sections:
        items = report["review"][key]
        lines.extend(["", f"{title} ({len(items)})", "-" * (len(title) + len(str(len(items))) + 3)])
        if not items:
            lines.append("None flagged.")
            continue
        for item in items[:TEXT_ITEMS_PER_SECTION]:
            if key == "short_speaker":
                lines.append(f"{item['speaker']}: {item['words']} words; check these appearances:")
                for appearance in item["appearances"]:
                    lines.append(f"  {timestamp(appearance['start'])}  {appearance['excerpt']}")
            else:
                at = timestamp(item["start"]) if item["start"] is not None else f"chunk {item['chunk_index']}"
                if key == "unknown_speaker":
                    detail = f"{item['words']} words"
                elif key == "low_word_score":
                    detail = f"{item['low_score_words']} low-score words; minimum {item['minimum_word_score']:.3f}"
                else:
                    detail = f"{item['speaker']}, similarity {item['cosine_similarity']:.3f} (threshold {item['threshold']:.2f})"
                lines.append(f"{at}  {detail}: {item['excerpt']}")
        if len(items) > TEXT_ITEMS_PER_SECTION:
            lines.append(f"... {len(items) - TEXT_ITEMS_PER_SECTION} more in quality_report.json")
    lines.append("")
    return "\n".join(lines)


def write_quality_report(
    output_dir: Path,
    *,
    duration_seconds: float | None = None,
    words: list[dict[str, Any]] | None = None,
    utterances: list[dict[str, Any]] | None = None,
    reconciliation: dict[str, Any] | None = None,
    chunks: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    state_path = output_dir / JOB_STATE_NAME
    if state_path.exists():
        state = _read_json(state_path)
        if state["status"] != "complete":
            raise ValueError("Job is incomplete. Resume transcription before generating the report.")
        if duration_seconds is None and "duration_seconds" in state:
            duration_seconds = float(state["duration_seconds"])
        if chunks is None and "chunks" in state:
            chunks = {int(chunk["index"]): chunk for chunk in state["chunks"]}

    words = words if words is not None else _read_json(output_dir / "words_with_speakers.json")
    utterances = utterances if utterances is not None else _read_json(output_dir / "transcript.json")
    reconciliation = reconciliation if reconciliation is not None else _read_json(output_dir / "speaker_reconciliation.json")
    if not isinstance(words, list) or not isinstance(utterances, list) or not isinstance(reconciliation, dict):
        raise ValueError("Completed-job JSON files have an unexpected structure")
    if chunks is None:
        chunks = _checkpoint_chunks(output_dir)
    if duration_seconds is None:
        duration_seconds = (
            max(float(chunk["core_end"]) for chunk in chunks.values())
            if chunks else max([float(item["end"]) for item in words] + [0.0])
        )
    report = build_quality_report(words, utterances, reconciliation, duration_seconds, chunks)
    _save_text_atomic(output_dir / "quality_report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _save_text_atomic(output_dir / "quality_report.txt", render_quality_report(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a quality review report from a completed job; no models or GPU required.")
    parser.add_argument("output", type=Path, help="Completed job directory, e.g. output/recording")
    args = parser.parse_args()
    try:
        output_dir = args.output.expanduser().resolve()
        with job_lock(output_dir):
            write_quality_report(output_dir)
    except (OSError, ValueError, KeyError, TypeError, JobBusyError) as error:
        parser.exit(1, f"Could not create quality report: {error}\n")
    print(f"Quality report: {args.output / 'quality_report.txt'}")
    print(f"Review data:    {args.output / 'quality_report.json'}")


if __name__ == "__main__":
    main()
