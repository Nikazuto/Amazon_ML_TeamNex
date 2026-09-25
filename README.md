# Business Entity Resolution — Amazon ML Challenge

End-to-end pipeline that matches Source 1 (reference) business records to
Source 2 / Source 3 (noisy) records using normalization, hybrid blocking,
engineered pairwise similarity features, a supervised classifier, and
F0.5-calibrated thresholding. No external APIs, geocoding, or business
registries are used anywhere in this pipeline.

## 1. Project layout

```
business_entity_resolution/
├── src/business_entity_resolution/
│   ├── __init__.py
│   ├── config.py            # single source of truth for all tunables (PipelineConfig)
│   ├── data_loader.py        # TSV ingestion + validation, source identification
│   ├── normalization.py      # name/address normalization
│   ├── validation_split.py   # leakage-safe S1-level train/val split
│   ├── blocking.py           # hybrid candidate generation (blocking)
│   ├── similarity.py         # low-level string-similarity primitives
│   ├── features.py           # pairwise feature extraction
│   ├── training.py           # positive/negative pair construction + model training
│   ├── evaluation.py         # blocking recall/reduction ratio + macro F0.5
│   ├── threshold.py          # F0.5 threshold sweep + decision policy
│   ├── inference.py          # candidate generation + scoring for a given dataset split
│   ├── output.py             # matching_results.tsv / candidate_pairs.tsv writers + validators
│   ├── experiment.py         # per-run config/metrics logging
│   └── pipeline.py           # orchestrates every stage end-to-end
├── utils/
│   └── validate_submission.py   # stdlib-only format validator (matches the challenge spec)
├── tests/                    # unit tests (not run automatically by this delivery)
├── experiments/              # run_<id>.json + results.csv get written here
├── models/                   # trained model + feature config get saved here
├── output/                   # matching_results.tsv / candidate_pairs.tsv get written here
├── dataset/
│   ├── train/                # <- put train_source1/2/3.tsv + train_ground_truth.tsv here
│   └── test/                 # <- put test_source1/2/3.tsv here
├── requirements.txt
├── run_pipeline.py           # single entry point
└── README.md
```

## 2. Setup

```bash
cd business_entity_resolution
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Pinned versions (`requirements.txt`):

```
pandas==2.2.2
numpy==1.26.4
scipy==1.13.1
scikit-learn==1.5.0
pyyaml==6.0.1
```

Only scikit-learn's `HistGradientBoostingClassifier` and `LogisticRegression`
are used as models — both BSD-3-Clause licensed, far under the 8B parameter
limit, and trained entirely on the supplied data (no pretrained weights, no
network calls).

## 3. Add the data

Copy the challenge dataset into `dataset/` so it looks like:

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

(`dataset/train/` and `dataset/test/` already exist as empty folders in this
package — just drop the seven `.tsv` files in.)

## 4. Run the full pipeline

```bash
python run_pipeline.py
```

This single command runs every stage in order — ingestion, normalization,
leakage-safe split, hybrid blocking, blocking-recall evaluation, feature
extraction, model training (logistic regression + gradient-boosted trees,
selected on validation macro F0.5), threshold calibration, singleton
handling, test-set inference, output writing, and finally the local
validator — and prints a summary plus the validator's PASS/FAIL result.

It produces:

```
output/matching_results.tsv
output/candidate_pairs.tsv
models/<trained model + feature config>
experiments/run_<id>.json
experiments/results.csv
```

### Useful flags

```bash
python run_pipeline.py --config my_config.json   # override any PipelineConfig field
python run_pipeline.py --data-dir dataset --output-dir output
python run_pipeline.py --skip-validator           # skip the automatic validator run
python run_pipeline.py --log-level DEBUG
```

A config file only needs to contain the fields you want to override, e.g.:

```json
{ "split": { "validation_fraction": 0.25 }, "blocking": { "ngram_top_k": 20 } }
```

## 5. Validate a submission manually

The pipeline runs this automatically, but it can also be run standalone
against any pair of output files (stdlib only, no dependencies):

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Prints `PASS` (exit 0) or a numbered list of issues (exit 1). This checks
*format* only — schema, ID validity, candidate/match consistency, coverage —
not match quality. Match quality is reported separately by the pipeline as
validation macro F0.5.

## 6. Tests

Unit tests live in `tests/` (TSV loading, source ID, normalization,
blocking/recall, feature extraction, F0.5 calculation, singleton handling,
output formatting, duplicate detection, matched-⊆-candidates invariant).
They are included for reference/audit but are **not run as part of this
delivery** — run them yourself with:

```bash
pip install pytest
pytest tests/ -v
```

## 7. Reproducibility

Every random source (Python `random`, NumPy, split seed, model
`random_state`) is seeded from `PipelineConfig.random_seed` (default 42) via
`seed_everything()`, called at the start of `run_full_pipeline`. The exact
configuration that produced a given `output/` pair is always reconstructable
from `experiments/run_<id>.json`.

## 8. Architecture summary

```
raw TSVs
  -> data_loader: ingestion, source-prefix validation, dedup/consistency checks
  -> normalization: name (case-fold, punctuation, legal-suffix, abbreviation,
       token-sort) + address (abbreviation, numeric/PIN extraction, tokens)
  -> validation_split: 80/20 split at the S1-entity level (no leakage),
       optional country-holdout proxy for the unseen-France generalization test
  -> blocking: UNION of exact-name, token-based, char-n-gram TF-IDF kNN,
       address-based, and fuzzy (name+address n-gram) candidate generation;
       country is never used as a hard filter (open-set, soft signal only)
  -> evaluation.evaluate_blocking: blocking recall + reduction ratio on the
       validation split, so blocking-stage misses are diagnosed separately
       from model/threshold misses
  -> features: name/address/country similarity features (Jaccard, TF-IDF
       cosine, char n-gram cosine, edit similarity, PIN/component overlap,
       missing-data indicators, interaction features)
  -> training: positives from ground truth, negatives sampled from blocking
       candidates (not the full cross-join); logistic regression and a
       gradient-boosted tree model are both trained and compared on the same
       leakage-free validation split, selected by macro F0.5
  -> threshold: sweep on validation only, optimizing macro F0.5 (precision-
       weighted); optional top1-vs-top2 margin rule, only if it improves F0.5
  -> inference: candidates generated for every test S1 entity, scored, and
       thresholded the same way; entities with no candidate above threshold
       get an explicit empty match list (singletons)
  -> output: matching_results.tsv + candidate_pairs.tsv, with matched IDs
       always a subset of that entity's candidate IDs
  -> utils/validate_submission.py: automatic format check before declaring
       the run complete
```

Country is only ever consumed as an open-set string feature (equality/soft
similarity), never as a blocking filter or a hard-coded enum, so the France
records in the test set are handled the same way as any other country.
