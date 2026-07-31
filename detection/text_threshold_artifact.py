"""Pure validation for text threshold artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

ValidationMode = Literal["runtime", "readiness"]
_MISSING = object()


@dataclass(frozen=True)
class ArtifactValidationResult:
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_text_threshold_artifact(
    payload: Any,
    *,
    mode: ValidationMode,
    default_low_threshold: float = 0.4,
    default_high_threshold: float = 0.8,
    expected_dataset_profile: str | None = None,
) -> ArtifactValidationResult:
    """Validate an artifact for compatible runtime use or strict v2 readiness."""
    errors: list[str] = []

    def add(*codes: str) -> None:
        for code in codes:
            if code not in errors:
                errors.append(code)

    if mode not in {"runtime", "readiness"}:
        raise ValueError(f"Unknown artifact validation mode: {mode}")
    strict = mode == "readiness"

    if not isinstance(payload, Mapping):
        return ArtifactValidationResult(("artifact_invalid_root",))

    schema_version = payload.get("artifact_schema_version", 1)
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in {1, 2}
    ):
        return ArtifactValidationResult(("artifact_unsupported_schema_version",))
    if strict and schema_version != 2:
        add("artifact_schema_version_must_be_2")

    if strict:
        if (
            not isinstance(expected_dataset_profile, str)
            or not expected_dataset_profile
            or payload.get("dataset_profile") != expected_dataset_profile
        ):
            add("artifact_dataset_profile_mismatch")
        root_target_fpr = payload.get("target_fpr", _MISSING)
        if root_target_fpr is _MISSING:
            add("artifact_target_fpr_missing")
        elif not _valid_target_fpr(root_target_fpr):
            add("artifact_target_fpr_invalid")
    else:
        root_target_fpr = payload.get("target_fpr", _MISSING)
        if root_target_fpr is not _MISSING and not _valid_target_fpr(
            root_target_fpr
        ):
            add(
                "artifact_invalid_group_diagnostics",
                "artifact_target_fpr_invalid",
            )

    groups = payload.get("groups")
    if not isinstance(groups, Mapping):
        add("artifact_invalid_groups")
        return ArtifactValidationResult(tuple(errors))
    if strict and not groups:
        add("artifact_groups_must_be_non_empty_mapping")

    for group_name, group in groups.items():
        if not isinstance(group_name, str) or not group_name:
            add("artifact_invalid_group_entry", "artifact_group_name_invalid")
            continue
        if not isinstance(group, Mapping):
            add("artifact_invalid_group_entry", "artifact_group_entry_invalid")
            continue

        low = group.get(
            "low_threshold",
            _MISSING if strict else default_low_threshold,
        )
        high = group.get(
            "high_threshold",
            _MISSING if strict else default_high_threshold,
        )
        if low is _MISSING:
            add(
                "artifact_invalid_thresholds",
                "artifact_group_low_threshold_missing",
            )
        elif not _unit_interval_number(low):
            add(
                "artifact_invalid_thresholds",
                "artifact_group_low_threshold_invalid",
            )
        if high is _MISSING:
            add(
                "artifact_invalid_thresholds",
                "artifact_group_high_threshold_missing",
            )
        elif not _unit_interval_number(high):
            add(
                "artifact_invalid_thresholds",
                "artifact_group_high_threshold_invalid",
            )
        if (
            low is not _MISSING
            and high is not _MISSING
            and _finite_number(low)
            and _finite_number(high)
            and float(low) > float(high)
        ):
            add(
                "artifact_invalid_thresholds",
                "artifact_group_thresholds_inverted",
            )

        if strict:
            floor = group.get("threshold_floor", _MISSING)
            if floor is _MISSING:
                add("artifact_group_threshold_floor_missing")
            elif not _unit_interval_number(floor):
                add("artifact_group_threshold_floor_invalid")
            elif (
                high is not _MISSING
                and _finite_number(high)
                and float(floor) > float(high)
            ):
                add("artifact_group_high_threshold_below_floor")

            group_target_fpr = group.get("target_fpr", _MISSING)
            if group_target_fpr is _MISSING:
                add("artifact_group_target_fpr_missing")
            elif not _valid_target_fpr(group_target_fpr):
                add("artifact_group_target_fpr_invalid")
            elif (
                root_target_fpr is not _MISSING
                and _valid_target_fpr(root_target_fpr)
                and not math.isclose(
                    float(group_target_fpr),
                    float(root_target_fpr),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                add("artifact_group_target_fpr_mismatch")

        _validate_counts(group, strict=strict, add=add)
        _validate_diagnostics(group, strict=strict, add=add)

    return ArtifactValidationResult(tuple(errors))


def _validate_counts(
    group: Mapping[str, Any],
    *,
    strict: bool,
    add: Any,
) -> None:
    optional_counts = (
        "human_calibration_samples",
        "human_validation_samples",
        "validation_samples",
        "ai_validation_samples",
        "validation_false_positives",
        "ai_true_positives",
        "human_samples",
    )
    required_counts = {
        "human_calibration_samples",
        "human_validation_samples",
        "ai_validation_samples",
        "validation_false_positives",
        "ai_true_positives",
    }
    for field_name in optional_counts:
        value = group.get(field_name, _MISSING)
        if value is _MISSING:
            if strict and field_name in required_counts:
                add(
                    "artifact_invalid_group_diagnostics",
                    f"artifact_group_{field_name}_missing",
                )
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or (
                strict
                and field_name == "human_calibration_samples"
                and value <= 0
            )
        ):
            add(
                "artifact_invalid_group_diagnostics",
                f"artifact_group_{field_name}_invalid",
            )

    human_validation = group.get("human_validation_samples")
    false_positives = group.get("validation_false_positives")
    if (
        _nonnegative_int(human_validation)
        and _nonnegative_int(false_positives)
        and false_positives > human_validation
    ):
        add(
            "artifact_invalid_group_diagnostics",
            "artifact_group_validation_false_positives_exceed_samples",
        )
    ai_validation = group.get("ai_validation_samples")
    true_positives = group.get("ai_true_positives")
    if (
        _nonnegative_int(ai_validation)
        and _nonnegative_int(true_positives)
        and true_positives > ai_validation
    ):
        add(
            "artifact_invalid_group_diagnostics",
            "artifact_group_ai_true_positives_exceed_samples",
        )


def _validate_diagnostics(
    group: Mapping[str, Any],
    *,
    strict: bool,
    add: Any,
) -> None:
    unit_fields = (
        "validation_fpr",
        "ai_recall",
        "ai_fnr",
        "validation_auroc",
    )
    interval_fields = (
        "validation_fpr_interval_95",
        "ai_recall_interval_95",
    )
    for field_name in unit_fields:
        value = group.get(field_name, _MISSING)
        if value is _MISSING:
            if strict:
                add(
                    "artifact_invalid_group_diagnostics",
                    f"artifact_group_{field_name}_missing",
                )
        elif value is not None and not _unit_interval_number(value):
            add(
                "artifact_invalid_group_diagnostics",
                f"artifact_group_{field_name}_invalid",
            )

    for field_name in interval_fields:
        value = group.get(field_name, _MISSING)
        if value is _MISSING:
            if strict:
                add(
                    "artifact_invalid_group_diagnostics",
                    f"artifact_group_{field_name}_missing",
                )
        elif value is not None and not _valid_interval(value):
            add(
                "artifact_invalid_group_diagnostics",
                f"artifact_group_{field_name}_invalid",
            )

    warnings = group.get("warnings", _MISSING)
    if warnings is _MISSING:
        if strict:
            add(
                "artifact_invalid_group_diagnostics",
                "artifact_group_warnings_missing",
            )
    elif not isinstance(warnings, list) or not all(
        isinstance(item, str) for item in warnings
    ):
        add(
            "artifact_invalid_group_diagnostics",
            "artifact_group_warnings_invalid",
        )

    reliability = group.get("reliability_status", _MISSING)
    if reliability is _MISSING:
        if strict:
            add(
                "artifact_invalid_group_diagnostics",
                "artifact_group_reliability_status_missing",
            )
    elif not isinstance(reliability, str) or not reliability.strip():
        add(
            "artifact_invalid_group_diagnostics",
            "artifact_group_reliability_status_invalid",
        )

    quantiles = group.get("calibration_quantiles", _MISSING)
    if strict and quantiles is _MISSING:
        add(
            "artifact_invalid_group_diagnostics",
            "artifact_group_calibration_quantiles_missing",
        )
    elif quantiles is not _MISSING and not isinstance(quantiles, Mapping):
        add(
            "artifact_invalid_group_diagnostics",
            "artifact_group_calibration_quantiles_invalid",
        )


def _finite_number(value: Any) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _unit_interval_number(value: Any) -> bool:
    return _finite_number(value) and 0.0 <= float(value) <= 1.0


def _valid_target_fpr(value: Any) -> bool:
    return _finite_number(value) and 0.0 < float(value) <= 1.0


def _nonnegative_int(value: Any) -> bool:
    return bool(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _valid_interval(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    low, high = value
    return (
        _unit_interval_number(low)
        and _unit_interval_number(high)
        and float(low) <= float(high)
    )
