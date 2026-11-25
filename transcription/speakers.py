"""Reconcile speaker identities across chunks and attribute words to speakers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from . import checkpoints, runtime

if TYPE_CHECKING:
    import numpy as np


def normalize_vector(v: np.ndarray) -> np.ndarray | None:
    if v.ndim != 1 or not runtime.np.all(runtime.np.isfinite(v)):
        return None
    norm = float(runtime.np.linalg.norm(v))
    if norm <= 1e-12:
        return None
    return v / norm


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return -1.0
    return float(runtime.np.clip(runtime.np.dot(a, b), -1.0, 1.0))


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
        data = checkpoints.load_json(checkpoint)
        turns = data["turns_extended"]
        raw_embeddings: dict[str, list[float]] = data.get("embeddings", {})
        core_durations: dict[str, float] = data.get("core_durations", {})

        local_labels = sorted(
            set(raw_embeddings.keys()) | {str(t["speaker"]) for t in turns}
        )
        local_vectors = {
            label: normalize_vector(runtime.np.asarray(raw_embeddings[label], dtype=runtime.np.float32))
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
                    "weighted_sum": vector.astype(runtime.np.float64) * weight,
                    "weight": weight,
                    "centroid": vector.astype(runtime.np.float64),
                }
            else:
                profile = global_profiles[global_id]
                profile["weighted_sum"] += vector.astype(runtime.np.float64) * weight
                profile["weight"] += weight
                centroid = normalize_vector(profile["weighted_sum"].astype(runtime.np.float32))
                if centroid is not None:
                    profile["centroid"] = centroid.astype(runtime.np.float64)

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
    checkpoints.save_json_atomic(output_path, report)
    print(f"Stable speakers: {len(global_profiles)}")
    print(f"Speaker reconciliation report: {output_path}")
    return final_core_turns


def save_single_speaker_reconciliation(path: Path) -> None:
    """Record the explicit user assumption without inventing diarization turns."""
    checkpoints.save_json_atomic(path, {
        "mode": "single_speaker",
        "global_speakers": ["SPEAKER_00"],
        "chunks": [],
        "speaker_turns": [],
    })


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
