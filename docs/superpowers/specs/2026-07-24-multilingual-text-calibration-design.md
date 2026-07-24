# Multilingual Multi-domain Text Calibration Design

Date: 2026-07-24

## Objective

Extend the existing false-positive controller from Chinese literature and
academic calibration to reproducible Chinese and English, multi-domain
calibration. The system must continue using the existing Yuchuan detectors,
control the held-out human false-positive rate, and additionally report AI
recall without training or modifying detector weights.

The bootstrap remains an offline preparation command. Runtime detection reads
the generated threshold artifact and never performs corpus similarity search.

## Scope

The first implementation includes:

- Existing Chinese poetry, classical literature, and CSL sources.
- HC3 Chinese and English human/ChatGPT answers.
- CLUE TNEWS Chinese human-written news.
- M4/SemEval multilingual, multi-domain human and machine texts, restricted to
  Chinese and English.
- RAID non-adversarial English human and machine texts.
- Chinese and English runtime routing for `literature`,
  `academic_technical`, `official_news`, and `general_qa`.
- Human FPR, AI recall/FNR, AUROC, sample counts, confidence intervals, and
  reliability warnings in the artifact and user-facing diagnostics.

The first implementation excludes:

- RAID adversarial variants.
- Languages other than Chinese and English.
- Fine-tuning a new detector.
- Online corpus lookup or semantic similarity matching.
- Automatic threshold replacement when required validation evidence is
  incomplete.

## Resource Strategy

Resources use official project or dataset distribution endpoints. Large
sources are read sequentially and normalized into project-local JSONL. The
bootstrap uses deterministic reservoir sampling and only sends selected
samples to the detector.

The default resource profile is `multidomain`:

| Source | Language | Role | Normalized domain |
|---|---|---|---|
| Chinese Poetry | zh | human | literature |
| Chinese Classical Corpus | zh | human | literature |
| CSL | zh | human | academic_technical |
| CLUE TNEWS | zh | human | official_news |
| HC3 Chinese | zh | human + AI | general_qa |
| HC3 English | en | human + AI | general_qa |
| M4/SemEval | zh + en | human + AI | mapped from source metadata |
| RAID non-adversarial | en | human + AI | mapped from domain metadata |

RAID adversarial data is not downloaded by the default profile. The source
registry records repository, revision when available, license, selected split,
content hash, and completion status.

## Canonical Sample Contract

Dataset-specific adapters emit a common immutable record:

```text
TextSample
  text: str
  label: human | ai
  language: zh | en
  domain: literature | academic_technical | official_news | general_qa
  source: str
  source_id: str
  generator: str | null
  split: str
```

Adapters reject empty text, unsupported languages, unknown labels, and records
without stable source identity. Deduplication uses normalized text hashes.
Paired human/AI records retain a shared source ID so split assignment happens
at pair level and cannot leak one side of a prompt into a different partition.

## Length and Sampling

Effective length removes whitespace. Samples shorter than
`min_effective_chars` are excluded from calibration and validation because
runtime detection reports them as insufficient evidence.

- `short`: `min_effective_chars <= length <= short_text_chars`
- `long`: `length > short_text_chars`

Long documents produce sentence-aware contiguous windows. The default remains
one window per document to prevent a few very large documents from dominating
the score distribution.

Reservoir sampling is deterministic and without replacement. Its seed, scan
limit, selected sample IDs, and bucket counts are recorded in the artifact.
The initial defaults per language/domain/length group are:

- 5,000 human calibration samples.
- 1,000 held-out human validation samples.
- 1,000 held-out AI validation samples.
- 500,000 source records scanned per dataset.

Smaller groups use all available records while preserving a held-out
partition. A group below the configured minimum evidence requirement does not
produce a calibrated threshold.

## Threshold and Evaluation Logic

Only human calibration scores determine the high threshold. For target FPR
`alpha`, the threshold is the finite-sample conformal upper quantile, bounded
below by the configured domain floor.

Held-out samples are never used to fit the threshold:

- Human validation computes false positives, FPR, and a Wilson 95% interval.
- AI validation computes true positives, recall, FNR, and a Wilson 95%
  interval.
- Combined held-out human and AI scores compute AUROC when both classes exist.

AI validation measures the cost of reducing false positives; it does not
silently lower a threshold that satisfies the human FPR constraint.

A group is marked `warning` when any of the following applies:

- Threshold is at or above the configured near-ceiling level.
- Held-out human FPR confidence interval is incompatible with the target.
- Human or AI validation samples are below the configured reliable minimum.
- AI recall is below the configured minimum.
- AUROC cannot be computed.

The artifact is written atomically only after every required source and group
passes validation. A failed run leaves the last valid artifact untouched.

## Runtime Routing

Language routing remains character based for `zh`, `en`, and `mixed`.
Domain routing moves into a dedicated bilingual classifier module with explicit
keyword evidence and confidence. It is not presented as semantic
classification.

Threshold lookup order is:

1. Exact `language:domain:length` calibrated group.
2. Exact `language:domain` calibrated group.
3. Configured domain fallback for that language and length.
4. Language-level calibrated fallback.
5. Built-in conservative fallback.

Mixed-language text uses a mixed group only when independently calibrated;
otherwise it uses the stricter applicable Chinese or English threshold and
reports a fallback warning.

## Modules and Files

### `detection/text_resources.py`

Owns resource specifications, download state, canonical sample iteration,
checksums, completion markers, and project-local paths.

### `detection/text_resource_adapters.py`

Owns HC3, TNEWS, M4/SemEval, RAID, and legacy source parsing. Adapters have no
model dependency and are tested with small local fixtures.

### `detection/text_domain.py`

Owns bilingual runtime domain routing and source-domain normalization. Dataset
metadata mapping and runtime text classification share the same canonical
domain names.

### `scripts/bootstrap_text_resources.py`

Owns model bootstrap, resource preparation, deterministic partitioning,
detector scoring, threshold fitting, held-out evaluation, progress output,
status/statistics profiles, and atomic artifact publication.

### `detection/false_positive_controller.py`

Consumes artifact version 2, performs exact multilingual threshold lookup, and
returns FPR/recall/reliability evidence with the decision.

### `config.yaml`

Defines resource profile, source enable flags, scan and sample limits, minimum
evidence, target FPR, minimum acceptable AI recall, and domain floors.

### UI, reports, dependencies, and documentation

- `api/dashboard.py` displays language, domain, threshold group, human FPR,
  AI recall, sample counts, and warnings.
- `reports/generator.py` includes the same evidence in generated reports.
- `requirements.txt` includes the dataset/parquet/progress dependencies.
- Linux deployment documentation describes disk requirements, commands,
  progress monitoring, resumability, and offline startup after bootstrap.

## Command Interface

```bash
python -m scripts.bootstrap_text_resources \
  --profile all \
  --dataset-profile multidomain \
  --force-calibration
```

Additional read-only profiles:

```bash
python -m scripts.bootstrap_text_resources --profile status
python -m scripts.bootstrap_text_resources --profile stats
```

`status` reports model, source, normalized cache, and artifact readiness.
`stats` reports counts by source, label, language, domain, and length bucket
without loading detector models.

## Failure Handling

- Downloads use partial files, timeouts, retries, and atomic rename.
- Completed source markers include the source specification fingerprint;
  changed source definitions invalidate stale markers.
- Individual malformed rows are counted and reported.
- Detector exceptions are summarized by model and exception type; they are not
  converted into a misleading "too few samples" error.
- Required-source failure prevents artifact publication.
- Optional-source failure is visible and only permits publication when every
  required language/domain group still meets minimum evidence.
- Interrupted runs reuse completed downloads and normalized files.

## Testing and Acceptance

Unit tests use local fixtures and fake detector scores; they never download
external datasets.

Acceptance requires:

- HC3 pair-level splitting prevents cross-partition leakage.
- Chinese and English groups are emitted with correct domain and length keys.
- Text below runtime minimum length is excluded.
- Sampling is deterministic for the same seed and changes for a different
  seed.
- Human-only scores fit thresholds; AI scores only evaluate them.
- FPR, recall, FNR, AUROC, confidence intervals, and warnings are correct.
- Failed scoring or incomplete required data cannot overwrite a valid artifact.
- English runtime requests select English calibrated groups.
- Existing Chinese controller and report tests remain compatible.
- Bootstrap status and stats work without loading models.

## Deployment Outcome

After one successful online bootstrap, the server contains project-local model
cache, normalized sampled corpora, source manifests, and artifact version 2.
Normal API and Dashboard startup can then run with Hugging Face offline mode.
No corpus or model training occurs during ordinary requests.
