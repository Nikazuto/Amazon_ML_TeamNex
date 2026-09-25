"""
data_loader.py
===============
Robust TSV ingestion with validation, per Requirements Analysis FR-1.1/1.2.

Responsibilities:
  * Load S1/S2/S3 + ground-truth TSVs with an explicit sep="\\t".
  * Verify required columns exist.
  * Infer/validate source from BOTH the entity_id prefix and the originating
    file -- never rely on a hard-coded source column (none exists).
  * Detect duplicate IDs, missing required fields, and prefix/file mismatches.
  * Never raise on messy-but-parseable rows; collect issues and continue,
    since the challenge's own test data (e.g. France records) must still
    load successfully.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .config import COL_ADDRESS, COL_COUNTRY, COL_ENTITY_ID, COL_NAME, COL_SOURCE

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_COUNTRY]
SOURCE_PREFIXES = {"S1": "source1", "S2": "source2", "S3": "source3"}


@dataclass
class LoadReport:
    file_path: str
    expected_source: str
    n_rows: int = 0
    n_duplicate_ids: int = 0
    n_missing_name: int = 0
    n_missing_address: int = 0
    n_missing_country: int = 0
    n_prefix_mismatch: int = 0
    issues: List[str] = field(default_factory=list)

    def is_clean(self) -> bool:
        return not self.issues


def _prefix_of(entity_id: str) -> Optional[str]:
    if not isinstance(entity_id, str):
        return None
    for prefix in SOURCE_PREFIXES:
        if entity_id.startswith(prefix + "-"):
            return prefix
    return None


def load_source_file(path: str, expected_source: str) -> "tuple[pd.DataFrame, LoadReport]":
    """
    Load one source TSV (S1, S2, or S3) with validation.

    expected_source: one of "S1", "S2", "S3" -- used to cross-check the
    entity_id prefix against the file the record came from.
    """
    report = LoadReport(file_path=path, expected_source=expected_source)

    if not os.path.exists(path):
        report.issues.append(f"file not found: {path}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS + [COL_SOURCE]), report

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=True)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        report.issues.append(f"missing required columns: {missing_cols}")
        for c in missing_cols:
            df[c] = ""

    report.n_rows = len(df)

    df[COL_ENTITY_ID] = df[COL_ENTITY_ID].astype(str)
    null_ids = df[COL_ENTITY_ID].isna() | (df[COL_ENTITY_ID].str.strip() == "") | (df[COL_ENTITY_ID].str.lower() == "nan")
    if null_ids.any():
        report.issues.append(f"{int(null_ids.sum())} rows with null/empty entity_id (dropped)")
        df = df[~null_ids].copy()

    dup_mask = df[COL_ENTITY_ID].duplicated(keep="first")
    report.n_duplicate_ids = int(dup_mask.sum())
    if report.n_duplicate_ids:
        report.issues.append(f"{report.n_duplicate_ids} duplicate entity_id values (kept first occurrence)")
        df = df[~dup_mask].copy()

    inferred_prefix = df[COL_ENTITY_ID].map(_prefix_of)
    mismatch = inferred_prefix != expected_source
    report.n_prefix_mismatch = int(mismatch.sum())
    if report.n_prefix_mismatch:
        report.issues.append(
            f"{report.n_prefix_mismatch} rows whose entity_id prefix does not match the "
            f"originating file (expected {expected_source}); these rows are dropped to "
            f"prevent source mixing"
        )
        df = df[~mismatch].copy()

    report.n_missing_name = int((df[COL_NAME].isna() | (df[COL_NAME].astype(str).str.strip() == "")).sum())
    report.n_missing_address = int((df[COL_ADDRESS].isna() | (df[COL_ADDRESS].astype(str).str.strip() == "")).sum())
    report.n_missing_country = int((df[COL_COUNTRY].isna() | (df[COL_COUNTRY].astype(str).str.strip() == "")).sum())

    df[COL_SOURCE] = expected_source
    df = df.reset_index(drop=True)
    return df, report


def load_ground_truth(path: str) -> pd.DataFrame:
    from .config import GT_COL_MATCHES, GT_COL_S1

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=True)
    for col in [GT_COL_S1, GT_COL_MATCHES]:
        if col not in df.columns:
            raise ValueError(f"ground truth file missing required column: {col}")
    df[GT_COL_MATCHES] = df[GT_COL_MATCHES].fillna("")
    df[GT_COL_S1] = df[GT_COL_S1].astype(str)
    return df.reset_index(drop=True)


def parse_id_list(cell: str) -> List[str]:
    if cell is None or (isinstance(cell, float)):
        return []
    cell = str(cell).strip()
    if cell == "" or cell.lower() == "nan":
        return []
    return [x.strip() for x in cell.split(",") if x.strip()]


@dataclass
class DatasetBundle:
    s1: pd.DataFrame
    s2: pd.DataFrame
    s3: pd.DataFrame
    ground_truth: Optional[pd.DataFrame]
    reports: Dict[str, LoadReport]

    def all_s2_s3_ids(self) -> set:
        return set(self.s2[COL_ENTITY_ID]) | set(self.s3[COL_ENTITY_ID])


def load_dataset(data_dir: str, split: str) -> DatasetBundle:
    """
    split: "train" or "test"
    """
    prefix = f"{split}_"
    base = os.path.join(data_dir, split)
    s1_path = os.path.join(base, f"{prefix}source1.tsv")
    s2_path = os.path.join(base, f"{prefix}source2.tsv")
    s3_path = os.path.join(base, f"{prefix}source3.tsv")
    gt_path = os.path.join(base, f"{prefix}ground_truth.tsv")

    s1, r1 = load_source_file(s1_path, "S1")
    s2, r2 = load_source_file(s2_path, "S2")
    s3, r3 = load_source_file(s3_path, "S3")

    reports = {"source1": r1, "source2": r2, "source3": r3}
    for name, r in reports.items():
        for issue in r.issues:
            logger.warning("[%s/%s] %s", split, name, issue)

    gt = None
    if split == "train" and os.path.exists(gt_path):
        gt = load_ground_truth(gt_path)
        # cross-check: every S1 id referenced in ground truth exists in s1
        s1_ids = set(s1[COL_ENTITY_ID])
        from .config import GT_COL_S1

        unknown = set(gt[GT_COL_S1]) - s1_ids
        if unknown:
            logger.warning("[train/ground_truth] %d source1_entity_id values not found in train_source1.tsv", len(unknown))

    return DatasetBundle(s1=s1, s2=s2, s3=s3, ground_truth=gt, reports=reports)
