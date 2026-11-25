"""Merge chunk words, build utterances, and write transcript formats."""

from __future__ import annotations

import html
import textwrap
import unicodedata
from pathlib import Path
from typing import Any

from . import audio, checkpoints


BOUNDARY_MATCH_SECONDS = 0.75
BOUNDARY_RECOVERY_SECONDS = 0.75


def timestamp_milliseconds(seconds: float) -> int:
    return max(0, int(round(seconds * 1000)))


def format_timestamp(seconds: float, srt: bool = False) -> str:
    return format_milliseconds(timestamp_milliseconds(seconds), srt=srt)


def format_milliseconds(ms: int, *, srt: bool = False) -> str:
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    sep = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


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
    previous_core = audio.in_core(word_midpoint(previous[1]), previous_chunk)
    current_core = audio.in_core(word_midpoint(current[1]), current_chunk)
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
        data = checkpoints.load_json(path)
        chunk_words = data["words_extended"]
        tagged = [(position, word) for position, word in enumerate(chunk_words)]
        extended.append(tagged)
        for position, word in tagged:
            if audio.in_core(word_midpoint(word), chunk):
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
                    and audio.in_core(previous_midpoint, previous_chunk)
                    and audio.in_core(current_midpoint, current_chunk)
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
            start = timestamp_milliseconds(item["start"])
            end = max(start + 1, timestamp_milliseconds(item["end"]))
            f.write(f"{format_milliseconds(start)} --> {format_milliseconds(end)}\n")
            f.write(f"{html.escape(wrapped, quote=False)}\n\n")
