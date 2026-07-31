from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research.text_attribution.audit import audit_collection
from research.text_attribution.production import build_plan_from_config, request_key
from research.text_attribution.run_config import load_run_config
from scripts.audit_attribution_data import main as audit_main


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "text_attribution" / "runs" / "smoke.json"


def _record(config, item, *, text: str, finish_reason: str, mismatch: bool) -> dict:
    family_by_model = {model.model_id: model.family_id for model in config.models}
    return {
        "run_id": config.run_id,
        "request_key": request_key(item),
        "family_id": family_by_model[item.model],
        "repeat_index": item.repeat_index,
        "request_model": item.model,
        "response_model": "different-route" if mismatch else item.model,
        "route_mismatch": mismatch,
        "text": text,
        "finish_reason": finish_reason,
        "prompt": {
            "prompt_id": item.prompt.prompt_id,
            "language": item.prompt.language,
            "domain": item.prompt.domain,
            "task_type": item.prompt.task_type,
            "target_length": item.prompt.target_length,
        },
        "parameters": {
            "temperature": item.parameters.temperature,
            "top_p": item.parameters.top_p,
            "max_tokens": item.parameters.max_tokens,
        },
    }


class AttributionAuditTests(unittest.TestCase):
    def test_audit_reports_coverage_quality_and_missing_cells(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        plan = build_plan_from_config(config)
        rows = [
            _record(config, plan[0], text="same", finish_reason="stop", mismatch=False),
            _record(config, plan[1], text="same", finish_reason="length", mismatch=True),
            _record(config, plan[2], text="", finish_reason="stop", mismatch=False),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            report = audit_collection(config, input_path=path)

        self.assertEqual(report["expected_requests"], 20)
        self.assertEqual(report["records"], 3)
        self.assertEqual(report["unique_successes"], 3)
        self.assertEqual(report["missing_requests"], 17)
        self.assertEqual(report["coverage_rate"], 0.15)
        self.assertEqual(report["empty_text_count"], 1)
        self.assertEqual(report["truncated_count"], 1)
        self.assertEqual(report["route_mismatch_count"], 1)
        self.assertEqual(report["duplicate_text_count"], 1)
        self.assertEqual(sum(report["counts_by_model"].values()), 3)
        self.assertEqual(sum(report["counts_by_family"].values()), 3)
        self.assertEqual(sum(report["counts_by_language"].values()), 3)
        self.assertEqual(sum(report["counts_by_task"].values()), 3)
        self.assertEqual(sum(report["counts_by_repeat"].values()), 3)
        self.assertEqual(len(report["missing_cells"]), 17)

    def test_audit_counts_error_records_without_treating_them_as_success(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        plan = build_plan_from_config(config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.jsonl"
            errors = root / "errors.jsonl"
            raw.write_text("", encoding="utf-8")
            errors.write_text(
                json.dumps(
                    {
                        "request_key": request_key(plan[0]),
                        "family_id": config.models[0].family_id,
                        "request_model": plan[0].model,
                        "http_status": 429,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = audit_collection(
                config,
                input_path=raw,
                error_path=errors,
            )

        self.assertEqual(report["records"], 0)
        self.assertEqual(report["error_records"], 1)
        self.assertEqual(report["errors_by_status"], {"429": 1})
        self.assertEqual(report["missing_requests"], 20)

    def test_audit_treats_max_tokens_finish_reason_as_truncated(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        plan = build_plan_from_config(config)
        row = _record(
            config,
            plan[0],
            text="partial",
            finish_reason="max_tokens",
            mismatch=False,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = audit_collection(config, input_path=path)

        self.assertEqual(report["truncated_count"], 1)

    def test_strict_cli_rejects_complete_but_route_mismatched_smoke(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        plan = build_plan_from_config(config)
        rows = [
            _record(
                config,
                item,
                text=f"unique-{index}",
                finish_reason="stop",
                mismatch=index == 0,
            )
            for index, item in enumerate(plan)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.jsonl"
            raw.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            exit_code = audit_main(
                [
                    "--config",
                    str(SMOKE_CONFIG),
                    "--input",
                    str(raw),
                    "--strict",
                ]
            )

        self.assertEqual(exit_code, 1)

    def test_minimum_model_coverage_accepts_usable_95_percent(self) -> None:
        config = load_run_config(SMOKE_CONFIG)
        plan = build_plan_from_config(config)
        rows = [
            _record(
                config,
                item,
                text=f"unique-{index}",
                finish_reason="stop",
                mismatch=False,
            )
            for index, item in enumerate(plan)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.jsonl"
            raw.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            ready_exit_code = audit_main(
                [
                    "--config",
                    str(SMOKE_CONFIG),
                    "--input",
                    str(raw),
                    "--minimum-model-coverage",
                    "0.95",
                ]
            )
            rows[0]["finish_reason"] = "length"
            raw.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            incomplete_exit_code = audit_main(
                [
                    "--config",
                    str(SMOKE_CONFIG),
                    "--input",
                    str(raw),
                    "--minimum-model-coverage",
                    "0.95",
                ]
            )

        self.assertEqual(ready_exit_code, 0)
        self.assertEqual(incomplete_exit_code, 1)


if __name__ == "__main__":
    unittest.main()
