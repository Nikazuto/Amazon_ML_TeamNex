# Blocking / Candidate Generation — Module Notes (Person B)

Scope: `src/business_entity_resolution/blocking.py` and its config
(`BlockingConfig` in `config.py`). This file documents the contract for
whoever consumes this module's output (Person C's `features.py`/
`training.py`/`inference.py`, and the shared `pipeline.py`), and how to run
this module's tests/evaluation on its own.

## 1. Contract

```python
def generate_candidates(
    s1_corpus: NormalizedCorpus,
    target_corpus: NormalizedCorpus,
    cfg: BlockingConfig,
) -> Dict[str, Set[str]]:
```

- **Input:** two `NormalizedCorpus` objects (build via `build_corpus(df,
  normalized_names, normalized_addresses)`), one for the Source 1 side, one
  for the combined Source 2 + Source 3 "target" side.
- **Output:** `{s1_entity_id: {candidate_target_id, ...}, ...}` — every id
  in `s1_corpus.ids` is guaranteed to be a key, even when its candidate set
  is empty. Every value is a `set()` of ids drawn only from
  `target_corpus.ids`.
- **This return value IS `candidate_pairs.tsv`'s content**, unmodified.
  `inference.py`'s `generate_and_score()` calls `generate_candidates()` and
  passes the exact same dict straight into `output.write_candidate_pairs()`
  — there is no separate/earlier candidate list that gets filtered later.
- `country` is never a parameter and never a hard filter (verified by
  `TestPublicInterface::test_no_hard_country_filtering`).
- The function is deterministic: identical `(s1_corpus, target_corpus,
  cfg)` always produces an identical result, independent of process-level
  hash randomization (see §3, bug #1).

## 2. How Person C consumes candidate pairs

1. **Never assume the candidate set is exhaustive.** Blocking is
   recall-oriented, not exhaustive — a target id absent from
   `candidates[s1_id]` was not retrieved by any of the six strategies and
   should be treated as "not scoreable," not "confidently a non-match."
   `evaluate_blocking()` (in `evaluation.py`) is how you find out how often
   that happens (`blocking_recall`), not something to infer from
   `features.py`.
2. **Only extract features for pairs that appear in the candidate set.**
   `inference.py::generate_and_score()` already does this correctly: it
   builds `pairs = [(s1_id, target_id) for s1_id, cand_set in
   candidates.items() for target_id in cand_set]` and only calls
   `extract_features_for_pairs()` on that list — never on a full
   cross-join. Follow the same pattern in any new call site.
3. **`matched_entity_ids` must be a subset of `candidate_entity_ids`.**
   This is enforced structurally in `output.py::check_decisions()` (an
   `OutputCheckReport` issue is raised otherwise) and is one of the
   challenge's own hard constraints (a matched id that was never a
   candidate is flagged as a pipeline bug). Any thresholding/decision logic
   should only ever select from `candidates[s1_id]`, never introduce an id
   from outside it.
4. **A missing S1 key is a bug, not a singleton.** Every S1 id always has
   an entry (possibly `set()`). If you ever see a `KeyError` looking up an
   S1 id in a `candidates` dict, that's an integration bug to report, not
   an implicit "treat as singleton" signal — singletons are represented as
   `set()`, present in the dict.
5. **Diagnosing false negatives:** `evaluation.diagnose_false_negative_sources(predicted, candidates, ground_truth)`
   already splits every missed ground-truth match into
   `blocking_false_negatives` (never a candidate — nothing Person C's model
   or threshold could have done) vs. `model_threshold_false_negatives` (was
   a candidate, scored, but not selected — Person C's stage). Use this
   before assuming a low F0.5 is a blocking problem or a model problem.
6. **Config is the only place strategies are tuned.** If a Person
   C experiment needs blocking behavior to change (e.g. wider candidate
   pools for a more powerful downstream classifier), change
   `BlockingConfig` fields (via `PipelineConfig.blocking` / a config JSON),
   not `blocking.py` itself.

## 3. Known limitations (flagged, not silently fixed)

1. **[FIXED] Candidate-cap tie-break was non-deterministic across
   processes.** When `len(candidates) > max_candidates_per_entity`, the cap
   keeps the `max_candidates_per_entity` targets whose normalized-name
   length is closest to the S1 name's length. Ties in that key used to be
   broken by Python's set iteration order, which depends on
   `PYTHONHASHSEED` — meaning the same input/config could silently produce
   a *different* capped candidate set across runs. Fixed by adding the
   candidate id itself as a deterministic secondary sort key. Covered by
   `TestCandidateCap::test_cap_is_deterministic_across_hash_seeds`.
2. **[FIXED] Address-token blocking hard-coded its minimum token length
   (3) instead of reading it from `BlockingConfig`,** unlike name-token
   blocking (`cfg.min_token_length`). Added `BlockingConfig.address_min_token_length`
   (default 3, preserving prior behavior) and wired it through both the
   index-build call and the S1-side loop. Covered by
   `TestAddressBlocking::test_address_min_token_length_is_configurable_not_hardcoded`.
3. **[OPEN, not changed]** The candidate-cap tie-break ranks purely by
   name-length closeness; it does not consider address similarity. On a
   pathologically common name with >`max_candidates_per_entity` matches,
   this could in principle drop a true match whose address is very similar
   but whose name length happens to differ more than a same-length
   near-duplicate's does. Not changed because (a) it would add a second
   algorithm/dependency to a safety-valve path that's supposed to be cheap
   and rarely triggered, and (b) there's no evidence from the (synthetic)
   test data that this actually costs recall in practice. If
   `evaluate_blocking()` on real data shows `n_true_matches_missed` pairs
   whose `s1_id` also has `candidate_count == max_candidates_per_entity`
   (i.e. capped), that's the signal to revisit this — log it in
   `experiments/blocking_baseline.md`.
4. **[OPEN, cross-module, flagged for Person A]** The postal-code blocking
   key comes from `NormalizedAddress.postal_code`, which
   `normalization.py` extracts via a bare `\b(\d{4,10}(?:-\d{2,4})?)\b`
   regex over the *raw* address — i.e. any 4-10 digit run, including a
   plain street number, can be picked up as a "postal code." This doesn't
   hurt recall (worst case it's an extra, harmless blocking key), but it
   means `postal_code`-based candidates aren't as strong a precision signal
   as the name suggests. Not a blocking.py bug and not changed here per the
   instruction not to touch Person A's normalization logic without cause;
   flagging so Person A/C know when interpreting `postal_code`-driven
   candidates or building features from it.

## 4. Commands

Run the blocking-specific unit tests:
```bash
pytest tests/test_blocking.py -v
```

Run the CLI-utility smoke tests:
```bash
pytest tests/test_evaluate_blocking_script.py -v
```

Run every test in the repo:
```bash
pytest tests/ -v
```

Quick blocking smoke test with no dataset files (built-in synthetic data):
```bash
python evaluate_blocking.py --synthetic
```

Full blocking evaluation against the real challenge dataset (needs
`dataset/train/train_*.tsv` populated):
```bash
python evaluate_blocking.py --data-dir dataset
```

Faster dev-loop pass over a large real dataset (sample S1 entities):
```bash
python evaluate_blocking.py --data-dir dataset --sample-size 2000
```

Try a config override and save a JSON report (append the numbers to
`experiments/blocking_baseline.md` by hand, or point future automation at
this same JSON):
```bash
python evaluate_blocking.py --data-dir dataset --config my_blocking_config.json \
    --output experiments/blocking_baseline_run.json
```

`my_blocking_config.json` can be either a bare `BlockingConfig`-shaped
object:
```json
{ "ngram_top_k": 20, "max_candidates_per_entity": 300 }
```
or a full `PipelineConfig`-shaped file (only the `blocking` section is
used):
```json
{ "blocking": { "ngram_top_k": 20 } }
```

Note: `evaluate_blocking.py` is intentionally blocking-only — it does not
train a model, so `--data-dir` runs report `blocking_recall`/
`n_true_matches_missed` (pure blocking-stage error analysis) but not
`model_threshold_false_negatives`, since there's no trained model at this
stage to make threshold decisions. For that split, run the full pipeline
(`run_pipeline.py`), which logs both.
