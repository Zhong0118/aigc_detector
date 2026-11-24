"""Optional text cleanup preserves evidence coordinates and calibration isolation."""
import json

import pytest

from detection.false_positive_controller import FalsePositiveController
from detection.providers import ApiFirstDetectionEngine
from detection.text_detector import RoutedTextAigcDetector, branch_scoring_protocol

BODY = ("正文讨论既有实验的局限[1]，并记录后续分析的证据。\n" * 33)
CHINESE = "\n参考文献\n" + "\n".join(
    f"［{i}］ 王某. 公开学术条目[J]. 样例学报, 2019, 5(2): 3-8。" for i in range(1, 5)
)
ENGLISH = "\nReferences\n" + "\n".join(
    f"[{i}] A. Author. Evaluation design. Journal of Examples, 2021, 4(2), 3–8." for i in range(1, 5)
)
TOC = "目　录\r\n" + "".join(f"第{i}章 研究设计………… {i}\r\n" for i in range(1, 5))


@pytest.mark.parametrize("tail", [CHINESE, ENGLISH])
def test_removes_verified_tail_only(tail):
    from detection.text_preprocessing import preprocess_text

    result = preprocess_text(BODY + tail)
    assert result.text.strip() == BODY.strip()
    assert result.excluded_ranges[0].kind == "references"
    assert (BODY+tail)[result.excluded_ranges[0].start:result.excluded_ranges[0].end].strip() == tail.strip()
    assert result.excluded_ranges[0].chars == sum(not c.isspace() for c in tail)


@pytest.mark.parametrize("protected", [
    BODY+"\n正文提到参考文献[4]，同时讨论四个变量。",
    BODY+CHINESE.replace("［3］", "［7］"),
    BODY+CHINESE+"\n附录\n这里是真实正文。",
    BODY+"\n参考文献\n"+"\n".join(f"[{i}] 这是2019年的一条正文，包含甲，乙，丙。" for i in range(1,5)),
    BODY+TOC,
])
def test_ambiguous_or_internal_material_stays(protected):
    from detection.text_preprocessing import preprocess_text

    result = preprocess_text(protected)
    assert result.text == protected
    assert not result.excluded_ranges


def test_toc_and_discontinuous_source_offsets():
    from detection.text_preprocessing import preprocess_text

    front = "研究计划\r\n"
    raw = front+TOC+BODY+CHINESE
    result = preprocess_text(raw)
    assert result.text.strip() == (front+BODY).strip()
    assert [r.kind for r in result.excluded_ranges] == ["table_of_contents", "references"]
    spans = result.source_ranges(0, 32)
    assert len(spans) == 2
    assert "".join(raw[a:b] for a,b in spans) == result.text[:32]


def test_disabled_keeps_calibrated_deployment_protocol(tmp_path):
    root="/home/zhongxu/projects/aigc_detector/models/text_binary"
    kwargs=dict(models={k:f"{root}/{m}" for k,m in {
        "zh_short":"20261117-mdeberta-v1", "zh_long":"20261123-mdeberta-v2-six-layer",
        "en_short":"20261124-en-v2","en_long":"20261124-en-v2"}.items()},
        mixed_strategy="dominant",short_text_chars=500,en_segment_target_chars=1400,
        en_segment_max_chars=1800,device="cpu",aggregation_mode="logit_mean",length_routing="document")
    raw=RoutedTextAigcDetector(**kwargs).scoring_protocol
    routes={"zh_short":"zh_short","zh_long":"zh_long","en_short":"en","en_long":"en"}
    temps={"zh_short":2.751766647048719,"zh_long":1.1264091564962642,"en":1.2728682333228396}
    payload=dict(raw_scoring_protocol=raw,model_routes=routes,
                 groups={k:{"temperature":v} for k,v in temps.items()},
                 scoring_protocol=branch_scoring_protocol(raw,routes,temps))
    artifact=tmp_path/"temperature.json"
    artifact.write_text(json.dumps(payload))
    implicit=RoutedTextAigcDetector(**kwargs,branch_calibration_path=str(artifact))
    explicit=RoutedTextAigcDetector(**kwargs,branch_calibration_path=str(artifact),preprocessing_enabled=False)
    assert explicit.scoring_protocol==implicit.scoring_protocol==payload["scoring_protocol"]=="token-logit-v1:2e084c9491123921"
    enabled=RoutedTextAigcDetector(**kwargs,branch_calibration_path=str(artifact),preprocessing_enabled=True)
    assert enabled.scoring_protocol != implicit.scoring_protocol
    assert not enabled.branch_temperatures
    engine=ApiFirstDetectionEngine()
    configured=engine._get_local_text_detector({"models":kwargs["models"], "aggregation_mode":"logit_mean",
                                               "length_routing":"document", "device":"cpu",
                                               "preprocessing":{"enabled":True}})
    assert configured.preprocessing_enabled


def test_enabled_routes_body_remaps_offsets_and_disables_legacy_cutoffs(monkeypatch):
    det=RoutedTextAigcDetector(
        models={"zh_short":"zh","zh_long":"zh","en_short":"en","en_long":"en"},
        mixed_strategy="dominant", short_text_chars=500, device="cpu", aggregation_mode="logit_mean",
        length_routing="document", preprocessing_enabled=True,
    )
    monkeypatch.setattr(det,"_run_model",lambda key,model,text:6.0)
    raw=TOC+BODY+ENGLISH
    out=det.detect(raw)
    assert out.details["selected_models"]==["zh_long"]
    assert [r["kind"] for r in out.details["excluded_ranges"]]==["table_of_contents","references"]
    for seg in out.details["segments"]:
        assert "".join(raw[a:b] for a,b in seg["source_ranges"])==seg["text"]
    out.status="ok"
    decision=FalsePositiveController().evaluate(raw,[out])
    assert decision.decision=="insufficient_evidence"
    assert decision.uncertainty_reasons==["scoring_protocol_not_evaluated"]
