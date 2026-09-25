#!/usr/bin/env python3
"""
validate_submission.py
=======================
Stdlib-only local validator mirroring the leaderboard's format checks
(problem statement "Output Format" / Requirements Analysis FR-6.1-6.4).

It checks FORMAT ONLY -- never match quality / F0.5 (Requirements
Analysis §10, AC-5: "passing validation says nothing about F0.5").

Usage:
    python3 utils/validate_submission.py \\
        --matching output/matching_results.tsv \\
        --candidate output/candidate_pairs.tsv \\
        --test-dir dataset/test

Exit code 0 + "PASS" if every check succeeds; exit code 1 + a numbered
issue list otherwise.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys


def read_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def read_source_ids(test_dir, filename, prefix):
    path = os.path.join(test_dir, filename)
    ids = set()
    if not os.path.exists(path):
        return ids, [f"test source file not found: {path}"]
    issues = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "entity_id" not in (reader.fieldnames or []):
            issues.append(f"{filename}: missing entity_id column")
            return ids, issues
        for row in reader:
            eid = (row.get("entity_id") or "").strip()
            if eid:
                ids.add(eid)
    return ids, issues


def parse_id_list(cell):
    cell = (cell or "").strip()
    if cell == "":
        return []
    return [x.strip() for x in cell.split(",") if x.strip() != ""]


def validate(matching_path, candidate_path, test_dir):
    issues = []

    if not os.path.exists(matching_path):
        return [f"matching file not found: {matching_path}"]
    if not os.path.exists(candidate_path):
        return [f"candidate file not found: {candidate_path}"]

    s1_ids, iss = read_source_ids(test_dir, "test_source1.tsv", "S1")
    issues += iss
    s2_ids, iss = read_source_ids(test_dir, "test_source2.tsv", "S2")
    issues += iss
    s3_ids, iss = read_source_ids(test_dir, "test_source3.tsv", "S3")
    issues += iss
    valid_target_ids = s2_ids | s3_ids

    if not s1_ids:
        issues.append("no S1 test ids could be loaded; aborting further checks")
        return issues

    m_fields, m_rows = read_tsv(matching_path)
    c_fields, c_rows = read_tsv(candidate_path)

    expected_m_fields = ["source1_entity_id", "matched_entity_ids"]
    expected_c_fields = ["source1_entity_id", "candidate_entity_ids"]
    if m_fields != expected_m_fields:
        issues.append(f"matching_results.tsv columns are {m_fields}, expected {expected_m_fields}")
    if c_fields != expected_c_fields:
        issues.append(f"candidate_pairs.tsv columns are {c_fields}, expected {expected_c_fields}")

    # --- matching_results.tsv checks ---
    seen_m_ids = set()
    dup_m_rows = set()
    matched_by_s1 = {}
    for row in m_rows:
        s1_id = (row.get("source1_entity_id") or "").strip()
        if s1_id in seen_m_ids:
            dup_m_rows.add(s1_id)
        seen_m_ids.add(s1_id)

        ids = parse_id_list(row.get("matched_entity_ids"))
        matched_by_s1[s1_id] = ids

        if s1_id not in s1_ids:
            issues.append(f"matching_results.tsv: row source1_entity_id '{s1_id}' not in test_source1.tsv")

        if len(ids) != len(set(ids)):
            issues.append(f"matching_results.tsv: duplicate ids within matched_entity_ids for {s1_id}")

        for mid in ids:
            if mid == s1_id or mid in s1_ids:
                issues.append(f"matching_results.tsv: self-match / S1 id '{mid}' in matched list for {s1_id}")
            elif mid not in valid_target_ids:
                issues.append(f"matching_results.tsv: matched id '{mid}' for {s1_id} not found in test S2/S3")

    if dup_m_rows:
        issues.append(f"matching_results.tsv: duplicate source1_entity_id rows: {sorted(dup_m_rows)[:10]}")

    missing_from_matching = s1_ids - seen_m_ids
    if missing_from_matching:
        issues.append(
            f"matching_results.tsv: {len(missing_from_matching)} test S1 entities missing "
            f"(e.g. {sorted(missing_from_matching)[:10]})"
        )

    extra_in_matching = seen_m_ids - s1_ids
    if extra_in_matching:
        issues.append(
            f"matching_results.tsv: {len(extra_in_matching)} rows reference S1 ids not in test set "
            f"(e.g. {sorted(extra_in_matching)[:10]})"
        )

    # --- candidate_pairs.tsv checks ---
    seen_c_ids = set()
    dup_c_rows = set()
    candidates_by_s1 = {}
    for row in c_rows:
        s1_id = (row.get("source1_entity_id") or "").strip()
        if s1_id in seen_c_ids:
            dup_c_rows.add(s1_id)
        seen_c_ids.add(s1_id)

        ids = parse_id_list(row.get("candidate_entity_ids"))
        candidates_by_s1[s1_id] = set(ids)

        if s1_id not in s1_ids:
            issues.append(f"candidate_pairs.tsv: row source1_entity_id '{s1_id}' not in test_source1.tsv")

        if len(ids) != len(set(ids)):
            issues.append(f"candidate_pairs.tsv: duplicate ids within candidate_entity_ids for {s1_id}")

        for cid in ids:
            if cid == s1_id or cid in s1_ids:
                issues.append(f"candidate_pairs.tsv: self/S1 id '{cid}' in candidate list for {s1_id}")
            elif cid not in valid_target_ids:
                issues.append(f"candidate_pairs.tsv: candidate id '{cid}' for {s1_id} not found in test S2/S3")

    if dup_c_rows:
        issues.append(f"candidate_pairs.tsv: duplicate source1_entity_id rows: {sorted(dup_c_rows)[:10]}")

    missing_from_candidates = s1_ids - seen_c_ids
    if missing_from_candidates:
        issues.append(
            f"candidate_pairs.tsv: {len(missing_from_candidates)} test S1 entities missing "
            f"(e.g. {sorted(missing_from_candidates)[:10]})"
        )

    # --- cross-file: matched subset of candidates ---
    for s1_id, matched_ids in matched_by_s1.items():
        cand_ids = candidates_by_s1.get(s1_id, set())
        not_in_cand = [mid for mid in matched_ids if mid not in cand_ids]
        if not_in_cand:
            issues.append(
                f"{s1_id}: matched ids not present in candidate_entity_ids (pipeline bug signal): {not_in_cand[:5]}"
            )

    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args()

    issues = validate(args.matching, args.candidate, args.test_dir)

    if not issues:
        print("PASS")
        sys.exit(0)
    else:
        print(f"FAIL ({len(issues)} issue(s)):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        sys.exit(1)


if __name__ == "__main__":
    main()
