"""
config.py
=========
Single source of truth for every tunable parameter in the pipeline.

Design goals (see Requirements Analysis / SRS FR-8.3, NFR-4):
  * Every stage reads its parameters from a `PipelineConfig` instance rather
    than hard-coding values, so a run is fully reconstructible from its
    config (needed for the experiment log).
  * The config is (de)serializable to/from JSON or YAML so a run's exact
    configuration can be persisted alongside its results.
  * Defaults live in one place, so `run_pipeline.py` can run with zero
    arguments and still be reproducible.

Nothing in this module hard-codes a country (see FR-1.3): "US"/"India"/etc.
never appear here.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


RANDOM_SEED_DEFAULT = 42


def seed_everything(seed: int = RANDOM_SEED_DEFAULT) -> None:
    """Make every source of randomness in the pipeline deterministic."""
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class SplitConfig:
    validation_fraction: float = 0.2
    split_seed: int = RANDOM_SEED_DEFAULT
    # Optional: hold out one training-only country entirely as a proxy for
    # the unseen-France generalization test (FR-1.3 / AC-8). None disables it.
    holdout_country_proxy: Optional[str] = None


@dataclass
class NormalizationConfig:
    # Legal-suffix / abbreviation dictionaries are defined in normalization.py;
    # this flag lets tests/experiments disable suffix stripping to measure its effect.
    strip_legal_suffixes: bool = True
    expand_abbreviations: bool = True
    char_ngram_n: int = 3


@dataclass
class BlockingConfig:
    # A. exact / token-sorted name blocking is always on (cheap, high precision)
    enable_exact_name_block: bool = True
    # B. token-based blocking
    enable_token_block: bool = True
    max_token_block_size: int = 400          # skip tokens that are too common (stopword-like)
    min_token_length: int = 3
    # C. character n-gram / TF-IDF nearest neighbor retrieval
    enable_ngram_block: bool = True
    ngram_top_k: int = 15
    ngram_char_range: tuple = (2, 4)
    # D. address-based blocking
    enable_address_block: bool = True
    address_top_k: int = 15
    # Minimum address-token length to be considered informative for address
    # token-blocking (mirrors min_token_length for names). Previously
    # hardcoded to 3 in blocking.py; exposed here per FR-8.3 so every run's
    # config is fully reconstructible from PipelineConfig.
    address_min_token_length: int = 3
    # E. combined name+address blocking is implicit in the union of the above
    # F. fuzzy retrieval via nearest neighbors on name+address combined n-grams
    enable_fuzzy_block: bool = True
    fuzzy_top_k: int = 15
    fuzzy_char_range: tuple = (2, 4)
    # Safety valve so a single S1 record with a very common name can't blow up
    # the candidate set / runtime.
    max_candidates_per_entity: int = 200


@dataclass
class NegativeSamplingConfig:
    negative_per_positive: int = 10
    seed: int = RANDOM_SEED_DEFAULT


@dataclass
class ModelConfig:
    models_to_try: List[str] = field(default_factory=lambda: ["logistic_regression", "gradient_boosted_trees"])
    logreg_C: float = 1.0
    logreg_max_iter: int = 2000
    gbt_max_depth: int = 6
    gbt_max_iter: int = 300
    gbt_learning_rate: float = 0.08
    gbt_l2_regularization: float = 0.1
    random_state: int = RANDOM_SEED_DEFAULT


@dataclass
class ThresholdConfig:
    thresholds_to_sweep: int = 199  # thresholds evenly spaced in (0, 1)
    min_threshold: float = 0.01
    max_threshold: float = 0.99
    use_margin_rule: bool = True   # optional top1-vs-top2 margin refinement (FR-5.2 / section 14)
    margin_min_gap: float = 0.05


@dataclass
class PipelineConfig:
    split: SplitConfig = field(default_factory=SplitConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    negative_sampling: NegativeSamplingConfig = field(default_factory=NegativeSamplingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    random_seed: int = RANDOM_SEED_DEFAULT
    data_dir: str = "dataset"
    output_dir: str = "output"
    models_dir: str = "models"
    experiments_dir: str = "experiments"

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return _tuples_to_lists(d)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PipelineConfig":
        d = copy.deepcopy(d)
        cfg = PipelineConfig()
        for section in ["split", "normalization", "blocking", "negative_sampling", "model", "threshold"]:
            if section in d and isinstance(d[section], dict):
                current = getattr(cfg, section)
                merged = {**dataclasses.asdict(current), **d[section]}
                if section == "blocking":
                    if "ngram_char_range" in merged and isinstance(merged["ngram_char_range"], list):
                        merged["ngram_char_range"] = tuple(merged["ngram_char_range"])
                    if "fuzzy_char_range" in merged and isinstance(merged["fuzzy_char_range"], list):
                        merged["fuzzy_char_range"] = tuple(merged["fuzzy_char_range"])
                setattr(cfg, section, type(current)(**merged))
        for key in ["random_seed", "data_dir", "output_dir", "models_dir", "experiments_dir"]:
            if key in d:
                setattr(cfg, key, d[key])
        return cfg

    def save(self, path: str) -> None:
        d = self.to_dict()
        if path.endswith((".yaml", ".yml")) and yaml is not None:
            with open(path, "w") as f:
                yaml.safe_dump(d, f, sort_keys=False)
        else:
            with open(path, "w") as f:
                json.dump(d, f, indent=2)

    @staticmethod
    def load(path: str) -> "PipelineConfig":
        if path.endswith((".yaml", ".yml")):
            if yaml is None:
                raise RuntimeError("pyyaml not installed; use a JSON config instead")
            with open(path) as f:
                d = yaml.safe_load(f)
        else:
            with open(path) as f:
                d = json.load(f)
        return PipelineConfig.from_dict(d or {})


def _tuples_to_lists(obj):
    if isinstance(obj, dict):
        return {k: _tuples_to_lists(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, list):
        return [_tuples_to_lists(v) for v in obj]
    return obj


# Canonical column names used throughout the pipeline.
COL_ENTITY_ID = "entity_id"
COL_NAME = "business_name"
COL_ADDRESS = "business_address"
COL_COUNTRY = "country"
COL_SOURCE = "source"

GT_COL_S1 = "source1_entity_id"
GT_COL_MATCHES = "matched_entity_ids"

OUT_COL_CANDIDATES = "candidate_entity_ids"
