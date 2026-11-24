"""Bind a verified portable bundle's calibration to its new, absolute model paths."""
import json
from pathlib import Path
from string import Template
import sys

import yaml

from detection.providers import ApiFirstDetectionEngine
from detection.text_detector import branch_scoring_protocol
from scripts.runtime_bundle import MODEL_NAMES, SOURCE_PROTOCOL


def configure(root: Path, device: str, api_port: str, ui_port: str) -> Path:
    assets=root/"runtime_assets"
    state=root/"runtime_state"
    template=(root/"configs/production_text_template.yaml").read_text(encoding="utf-8")
    # Replace markers after YAML parsing, preserving valid paths containing quotes.
    cfg=yaml.safe_load(template)

    def resolve(value):
        if isinstance(value,dict):
            return {k:resolve(v) for k,v in value.items()}
        if isinstance(value,str):
            return Template(value).substitute(APP_ROOT=str(root),DEVICE=device,API_PORT=api_port,UI_PORT=ui_port)
        return value
    cfg=resolve(cfg)
    cfg["api"]["port"]=int(api_port)
    cfg["dashboard"]["port"]=int(ui_port)
    if device=="cuda:0":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the selected venv")

    temp,thresholds=[json.loads((assets/"calibration"/file).read_text(encoding="utf-8"))
                     for file in ("temperature.json","thresholds.json")]
    for item in (temp,thresholds):
        if item["scoring_protocol"]!=SOURCE_PROTOCOL:
            raise ValueError("Unexpected source calibration")
    routes=temp["model_routes"]
    expected={"zh_short":"zh_short","zh_long":"zh_long","en_long":"en","en_short":"en"}
    if routes!=expected or thresholds["model_routes"]!=expected or set(thresholds["groups"])!=set(expected.values()):
        raise ValueError("Unexpected portable calibration routes")
    loc=cfg["detection"]["local_models"]["text_yuchuan"]
    if loc["models"]!={k:str(assets/"models/text_binary"/name) for k,name in MODEL_NAMES.items()}:
        raise ValueError("Template and verified checkpoint paths disagree")

    target=state/"production-text.yaml"
    target.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding="utf-8")
    runtime=ApiFirstDetectionEngine(str(target))
    raw=runtime._get_local_text_detector(dict(loc,branch_calibration_path=None)).scoring_protocol
    applied=branch_scoring_protocol(raw,routes,{k:v["temperature"] for k,v in temp["groups"].items()})
    temp.update(raw_scoring_protocol=raw,scoring_protocol=applied)
    thresholds["scoring_protocol"]=applied
    for file,value in (("temperature",temp),("thresholds",thresholds)):
        (state/f"{file}.json").write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    # Verify the same loader and controller used by the request path.
    runtime=ApiFirstDetectionEngine(str(target))
    det=runtime._get_local_text_detector(loc)
    if (det.scoring_protocol!=applied
            or runtime._text_fp_controller.artifact.get("scoring_protocol")!=applied
            or set(det.branch_temperatures)!=set(expected.values())):
        raise RuntimeError("Portable calibration preflight failed")
    return target


if __name__=="__main__":
    print(configure(Path(sys.argv[1]),*sys.argv[2:]))
