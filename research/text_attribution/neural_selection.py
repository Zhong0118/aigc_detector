"""Validation-only selection of completed neural attribution experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any


_EXPERIMENT_IDS = frozenset({"N0", "N1", "N2"})
_FINGERPRINT = re.compile(r"[0-9a-fA-F]{64}")
_VALIDATION_KEYS = frozenset(
    {
        "overall",
        "per_family",
        "confusion_matrix",
        "per_model",
        "calibration",
        "subgroups",
    }
)
_SELECTION_KEYS = frozenset({"selected_experiment", "criterion"})


@dataclass(frozen=True)
class NeuralExperimentScore:
    """One candidate's validation metrics used for deterministic ranking."""

    experiment_id: str
    source: str
    macro_f1: float
    worst_family_f1: float
    negative_log_likelihood: float

    @property
    def nll(self) -> float:
        """Short alias for the validation negative log likelihood."""

        return self.negative_log_likelihood

    def to_dict(self) -> dict[str, str | float]:
        return {
            "experiment_id": self.experiment_id,
            "source": self.source,
            "macro_f1": self.macro_f1,
            "worst_family_f1": self.worst_family_f1,
            "negative_log_likelihood": self.negative_log_likelihood,
        }


@dataclass(frozen=True)
class NeuralSelectionSummary:
    """Immutable, JSON-exportable result of neural experiment screening."""

    dataset_fingerprint: str
    split_manifest: str | None
    split_summary: tuple[tuple[str, int], ...]
    family_label_map: tuple[tuple[str, int], ...]
    ranking: tuple[NeuralExperimentScore, ...]
    top_two: tuple[str, ...]
    selection_scope: str = "validation_only"
    criterion: str = (
        "validation_macro_f1_desc,worst_family_f1_desc,"
        "negative_log_likelihood_asc,experiment_id_asc"
    )

    def to_dict(self) -> dict[str, Any]:
        """Return built-in JSON-safe values without exposing mutable internals."""

        return {
            "selection_scope": self.selection_scope,
            "criterion": self.criterion,
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_manifest": self.split_manifest,
            "split_summary": dict(self.split_summary),
            "family_label_map": dict(self.family_label_map),
            "ranking": [item.to_dict() for item in self.ranking],
            "top_two": list(self.top_two),
        }

    def as_dict(self) -> dict[str, Any]:
        """Compatibility alias for callers that use ``as_dict`` naming."""

        return self.to_dict()


@dataclass(frozen=True)
class _Candidate:
    score: NeuralExperimentScore
    dataset_fingerprint: str
    split_manifest: str | None
    split_summary: tuple[tuple[str, int], ...]
    family_label_map: tuple[tuple[str, int], ...]


def select_neural_experiments(
    paths: Sequence[Path],
) -> NeuralSelectionSummary:
    """Rank formal N0/N1/N2 runs using validation metrics and nothing else."""

    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise TypeError("paths must be a sequence of Path values")
    if not paths:
        raise ValueError("at least one neural experiment directory is required")

    candidates = tuple(_load_candidate(path) for path in paths)
    experiment_ids = [candidate.score.experiment_id for candidate in candidates]
    duplicate_ids = sorted(
        experiment_id
        for experiment_id in set(experiment_ids)
        if experiment_ids.count(experiment_id) > 1
    )
    if duplicate_ids:
        raise ValueError(
            "duplicate experiment ID(s): " + ", ".join(duplicate_ids)
        )

    reference = candidates[0]
    for candidate in candidates[1:]:
        if candidate.dataset_fingerprint != reference.dataset_fingerprint:
            raise ValueError("dataset fingerprint mismatch across candidates")
        if candidate.split_manifest != reference.split_manifest:
            raise ValueError("split manifest mismatch across candidates")
        if candidate.split_summary != reference.split_summary:
            raise ValueError("split summary mismatch across candidates")
        if candidate.family_label_map != reference.family_label_map:
            raise ValueError("family label map mismatch across candidates")

    ranking = tuple(
        candidate.score
        for candidate in sorted(
            candidates,
            key=lambda candidate: (
                -candidate.score.macro_f1,
                -candidate.score.worst_family_f1,
                candidate.score.negative_log_likelihood,
                candidate.score.experiment_id,
            ),
        )
    )
    return NeuralSelectionSummary(
        dataset_fingerprint=reference.dataset_fingerprint,
        split_manifest=reference.split_manifest,
        split_summary=reference.split_summary,
        family_label_map=reference.family_label_map,
        ranking=ranking,
        top_two=tuple(item.experiment_id for item in ranking[:2]),
    )


def _load_candidate(path: Path) -> _Candidate:
    if not isinstance(path, Path):
        raise TypeError("every candidate path must be a Path")
    if not path.exists():
        raise FileNotFoundError(f"candidate directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"candidate path must be a directory: {path}")

    run_config_path = path / "run_config.json"
    metrics_path = path / "metrics.json"
    run_config = _load_json_object(run_config_path, "run_config")
    metrics = _load_json_object(metrics_path, "metrics")

    experiment_id = _required_experiment_id(run_config)
    if run_config.get("smoke_test") is not False:
        raise ValueError(f"{path}: smoke-test candidates are not eligible")
    if run_config.get("evaluate_test") is not False:
        raise ValueError(f"{path}: test evaluation candidates are not eligible")

    dataset_fingerprint = _dataset_fingerprint(run_config, path)
    split_manifest = _split_manifest(run_config, path)
    split_summary = _split_summary(run_config, path)
    family_label_map = _family_label_map(run_config, path)
    validation = _validation_metrics(metrics, experiment_id, path)
    macro_f1, worst_family_f1, nll = _selection_metrics(
        validation,
        family_label_map,
        _validation_split_support(split_summary, path),
        path,
        experiment_id,
    )
    return _Candidate(
        score=NeuralExperimentScore(
            experiment_id=experiment_id,
            source=str(path.resolve()),
            macro_f1=macro_f1,
            worst_family_f1=worst_family_f1,
            negative_log_likelihood=nll,
        ),
        dataset_fingerprint=dataset_fingerprint,
        split_manifest=split_manifest,
        split_summary=split_summary,
        family_label_map=family_label_map,
    )


def _load_json_object(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _required_experiment_id(run_config: Mapping[str, Any]) -> str:
    experiment_id = run_config.get("experiment")
    if experiment_id not in _EXPERIMENT_IDS:
        raise ValueError("run_config experiment must be exactly one of N0, N1, N2")
    return experiment_id


def _dataset_fingerprint(
    run_config: Mapping[str, Any],
    source: Path,
) -> str:
    explicit = run_config.get("dataset_fingerprint")
    split_dataset = run_config.get("split_dataset_fingerprint")
    if explicit is not None and split_dataset is not None and explicit != split_dataset:
        raise ValueError(
            f"{source}: dataset fingerprint fields disagree"
        )
    value = explicit if explicit is not None else split_dataset
    return _validate_fingerprint(value, "dataset_fingerprint", source)


def _validate_fingerprint(value: Any, key: str, source: Path) -> str:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        readable = key.replace("_", " ")
        raise ValueError(
            f"{source}: {readable} must be a 64-character SHA256 hex string"
        )
    return value.lower()


def _split_manifest(
    run_config: Mapping[str, Any],
    source: Path,
) -> str | None:
    value = run_config.get("split_manifest")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: split manifest path must be a non-empty string")
    return value


def _split_summary(
    run_config: Mapping[str, Any],
    source: Path,
) -> tuple[tuple[str, int], ...]:
    value = run_config.get("splits")
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: split summary must be an object")
    normalized: list[tuple[str, int]] = []
    for split, support in value.items():
        if not isinstance(split, str) or not split:
            raise ValueError(f"{source}: split summary labels must be non-empty")
        if isinstance(support, bool) or not isinstance(support, int) or support < 0:
            raise ValueError(
                f"{source}: split summary support must be nonnegative integers"
            )
        normalized.append((split, support))
    return tuple(sorted(normalized))


def _validation_split_support(
    split_summary: tuple[tuple[str, int], ...],
    source: Path,
) -> int:
    split_counts = dict(split_summary)
    if "validation" not in split_counts:
        raise ValueError(
            f"{source}: run_config split summary must contain validation"
        )
    return split_counts["validation"]


def _family_label_map(
    run_config: Mapping[str, Any],
    source: Path,
) -> tuple[tuple[str, int], ...]:
    raw = run_config.get("family_to_id")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{source}: family label map must be a non-empty object")
    normalized: list[tuple[str, int]] = []
    for family, identifier in raw.items():
        if not isinstance(family, str) or not family:
            raise ValueError(f"{source}: family label map labels must be non-empty")
        if isinstance(identifier, bool) or not isinstance(identifier, int):
            raise ValueError(f"{source}: family label map IDs must be integers")
        normalized.append((family, identifier))
    normalized.sort(key=lambda item: item[1])
    expected_ids = list(range(len(normalized)))
    if [identifier for _, identifier in normalized] != expected_ids:
        raise ValueError(
            f"{source}: family label map IDs must be unique and contiguous from zero"
        )

    inverse = run_config.get("id_to_family")
    if inverse is not None:
        expected_inverse = {
            str(identifier): family for family, identifier in normalized
        }
        if inverse != expected_inverse:
            raise ValueError(f"{source}: family label map inverse is inconsistent")
    return tuple(normalized)


def _validation_metrics(
    metrics: Mapping[str, Any],
    experiment_id: str,
    source: Path,
) -> Mapping[str, Any]:
    experiment_keys = tuple(key for key in metrics if key != "selection")
    if experiment_keys != (experiment_id,):
        raise ValueError(
            f"{source}: metrics must contain exactly experiment {experiment_id}"
        )
    experiment_metrics = metrics.get(experiment_id)
    if not isinstance(experiment_metrics, Mapping):
        raise ValueError(f"{source}: experiment metrics must be an object")
    if "test" in experiment_metrics:
        raise ValueError(f"{source}: test metrics are forbidden during screening")
    if set(experiment_metrics) != {"validation"}:
        raise ValueError(
            f"{source}: metrics must contain validation and no other split"
        )
    validation = experiment_metrics.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError(f"{source}: validation metrics must be an object")
    if set(validation) != _VALIDATION_KEYS:
        raise ValueError(
            f"{source}: validation metric keys must be exactly "
            f"{sorted(_VALIDATION_KEYS)}"
        )
    _reject_test_keys(validation, f"{source}: validation")

    selection = metrics.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError(f"{source}: selection metadata must be an object")
    if set(selection) != _SELECTION_KEYS:
        raise ValueError(
            f"{source}: selection metadata keys must be exactly "
            f"{sorted(_SELECTION_KEYS)}"
        )
    _reject_test_keys(selection, f"{source}: selection")
    selected = selection["selected_experiment"]
    if selected != experiment_id:
        raise ValueError(
            f"{source}: selection metadata does not match {experiment_id}"
        )
    criterion = selection["criterion"]
    if not isinstance(criterion, str) or not criterion.strip():
        raise ValueError(f"{source}: selection criterion must be a non-empty string")
    return validation


def _reject_test_keys(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and (
                key.casefold() == "test" or key.casefold().startswith("test_")
            ):
                raise ValueError(f"{path} contains forbidden test data at {key}")
            _reject_test_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_test_keys(item, f"{path}[{index}]")


def _selection_metrics(
    row: Mapping[str, Any],
    family_label_map: tuple[tuple[str, int], ...],
    validation_split_support: int,
    source: Path,
    experiment_id: str,
) -> tuple[float, float, float]:
    prefix = f"{source}: metrics.{experiment_id}.validation"
    overall = _required_mapping(row, "overall", prefix)
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        _bounded_metric(overall, metric, f"{prefix}.overall", upper=1.0)
    macro_f1 = _bounded_metric(
        overall,
        "macro_f1",
        f"{prefix}.overall",
        upper=1.0,
    )
    overall_support = _support(overall, f"{prefix}.overall")
    if overall_support <= 0:
        raise ValueError(f"{prefix}.overall.support must be positive")

    calibration = _required_mapping(row, "calibration", prefix)
    _bounded_metric(
        calibration,
        "expected_calibration_error",
        f"{prefix}.calibration",
        upper=1.0,
    )
    _bounded_metric(
        calibration,
        "multiclass_brier_score",
        f"{prefix}.calibration",
        upper=2.0,
    )
    nll = _bounded_metric(
        calibration,
        "negative_log_likelihood",
        f"{prefix}.calibration",
        upper=None,
    )

    per_family = _required_mapping(row, "per_family", prefix)
    expected_labels = {family for family, _ in family_label_map}
    if set(per_family) != expected_labels:
        raise ValueError(
            f"{prefix}.per_family labels must match the family label map"
        )
    family_f1_values: list[float] = []
    family_support = 0
    for family, _ in family_label_map:
        diagnostics = _required_mapping(
            per_family,
            family,
            f"{prefix}.per_family",
        )
        for metric in ("precision", "recall", "f1-score"):
            value = _bounded_metric(
                diagnostics,
                metric,
                f"{prefix}.per_family.{family}",
                upper=1.0,
            )
            if metric == "f1-score":
                family_f1_values.append(value)
        family_support += _support(
            diagnostics,
            f"{prefix}.per_family.{family}",
        )
    if overall_support != family_support:
        raise ValueError(
            f"{prefix} support totals are inconsistent"
        )
    if overall_support != validation_split_support:
        raise ValueError(
            f"{prefix} validation split support does not match "
            "run_config split summary"
        )
    return macro_f1, min(family_f1_values), nll


def _required_mapping(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}.{key} must be an object")
    return value


def _bounded_metric(
    mapping: Mapping[str, Any],
    key: str,
    path: str,
    *,
    upper: float | None,
) -> float:
    if key not in mapping:
        raise ValueError(f"{path} is missing required metric {key}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}.{key} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{path}.{key} must be finite")
    if normalized < 0.0 or (upper is not None and normalized > upper):
        bound = f"[0, {upper:g}]" if upper is not None else "nonnegative"
        raise ValueError(f"{path}.{key} must be within {bound}")
    return normalized


def _support(mapping: Mapping[str, Any], path: str) -> int:
    if "support" not in mapping:
        raise ValueError(f"{path} is missing required support")
    value = mapping["support"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path}.support must be a nonnegative integer")
    return value
