# Blocking Baseline — Experiment Log

Template for recording blocking-stage experiments (Person B). One entry per
run of `evaluate_blocking.py` (or per `run_pipeline.py` run, reading the
`blocking_*` fields out of `experiments/run_<id>.json`). Copy the block
below and fill it in; keep the newest entry at the top.

Purpose: blocking determines the pipeline's recall ceiling (FR-3.2). This
log exists so a drop or improvement in blocking recall, reduction ratio, or
runtime is diagnosable against a specific config change, rather than
rediscovered from scratch each time.

---

## Run: `<run_id or date>`

**Command:**
```bash
python evaluate_blocking.py --data-dir dataset --sample-size <N or "full">
```

**Config (BlockingConfig fields changed from default, if any):**
| Field | Value |
|---|---|
| enable_exact_name_block | true |
| enable_token_block | true |
| min_token_length | 3 |
| enable_ngram_block | true |
| ngram_top_k | 15 |
| ngram_char_range | (2, 4) |
| enable_address_block | true |
| address_top_k | 15 |
| address_min_token_length | 3 |
| enable_fuzzy_block | true |
| fuzzy_top_k | 15 |
| fuzzy_char_range | (2, 4) |
| max_candidates_per_entity | 200 |

**Metrics:**
| Metric | Value |
|---|---|
| blocking_recall | |
| reduction_ratio | |
| total_candidates_generated | |
| avg_candidates_per_entity | |
| median_candidates_per_entity | |
| max_candidates_per_entity | |
| n_entities_zero_candidates | |
| n_true_matches | |
| n_true_matches_recovered | |
| n_true_matches_missed | |
| runtime_seconds | |
| runtime_seconds_per_1k_s1_entities | |

**Error analysis (blocking-stage misses):**
- Total missed: `<n_true_matches_missed>`
- Sample inspected (from `missed_pairs_sample`):
  - `<S1 id>` → `<target id>`: `<one-line note on why blocking missed it — e.g. no shared tokens, name too short for n-gram vocab, address entirely missing>`
- Pattern observed (if any): `<e.g. "misses cluster on records with empty business_address">`

**Notes / decision:**
- `<What changed since the last entry, and why. If this run's config was adopted as the new default, say so and update config.py's BlockingConfig defaults + this template's "Config" table above.>`

---

## Run: `<previous run, older at the bottom>`
...
