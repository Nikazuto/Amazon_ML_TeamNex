"""
output.py
=========
Writes the two required TSV artifacts (problem statement "Output Format";
SRS FR-6.1-6.4) and checks the invariants the leaderboard validator will
also check, so failures are caught locally before ever running the
official validator.

Invariants enforced here:
  * exactly one row per S1 entity supplied
  * matched_entity_ids / candidate_entity_ids only reference ids in
    `valid_target_ids` (i.e. S2/S3 ids that exist in the relevant split)
  * no duplicate ids within a single row's list
  * no self-matches (an S1 id can never appear in its own match list --
    guaranteed structurally here since valid_target_ids excludes S1 ids)
  * matched_entity_ids is always a subset of candidate_entity_ids for that
    same S1 entity
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Set

import pandas as pd


@dataclass
class OutputCheckReport:
    ok: bool
    issues: List[str] = field(default_factory=list)


def _format_id_list(ids: Set[str]) -> str:
    return ",".join(sorted(ids))


def check_decisions(
    s1_ids: List[str],
    matches: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
    valid_target_ids: Set[str],
) -> OutputCheckReport:
    issues: List[str] = []
    s1_id_set = set(s1_ids)

    missing_s1 = s1_id_set - set(matches.keys())
    if missing_s1:
        issues.append(f"{len(missing_s1)} S1 entities missing from matches (e.g. {sorted(missing_s1)[:5]})")

    for s1_id in s1_ids:
        matched = matches.get(s1_id, set())
        cand = candidates.get(s1_id, set())

        invalid_targets = matched - valid_target_ids
        if invalid_targets:
            issues.append(f"{s1_id}: matched ids not in valid target set: {sorted(invalid_targets)[:5]}")

        if s1_id in matched:
            issues.append(f"{s1_id}: self-match detected")

        not_in_candidates = matched - cand
        if not_in_candidates:
            issues.append(
                f"{s1_id}: matched ids not present in candidate_pairs: {sorted(not_in_candidates)[:5]}"
            )

    return OutputCheckReport(ok=(len(issues) == 0), issues=issues)


def write_matching_results(path: str, s1_ids: List[str], matches: Dict[str, Set[str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [{"source1_entity_id": s1_id, "matched_entity_ids": _format_id_list(matches.get(s1_id, set()))}
            for s1_id in s1_ids]
    df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    df.to_csv(path, sep="\t", index=False)


def write_candidate_pairs(path: str, s1_ids: List[str], candidates: Dict[str, Set[str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [{"source1_entity_id": s1_id, "candidate_entity_ids": _format_id_list(candidates.get(s1_id, set()))}
            for s1_id in s1_ids]
    df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    df.to_csv(path, sep="\t", index=False)


def write_outputs(
    output_dir: str,
    s1_ids: List[str],
    matches: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
    valid_target_ids: Set[str],
) -> OutputCheckReport:
    """Validates invariants, then writes both files regardless (issues are
    reported so the caller can decide whether to halt), matching the
    pipeline's "always fix and re-check" workflow around the official
    validator."""
    report = check_decisions(s1_ids, matches, candidates, valid_target_ids)
    write_matching_results(os.path.join(output_dir, "matching_results.tsv"), s1_ids, matches)
    write_candidate_pairs(os.path.join(output_dir, "candidate_pairs.tsv"), s1_ids, candidates)
    return report
