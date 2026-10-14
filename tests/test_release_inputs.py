"""Release inputs must remain usable without private server state."""
from pathlib import Path
import json
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_credentials_are_environment_variable_names():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    values = [
        config["detection"]["api_providers"]["hive"]["api_key_env"],
        config["detection"]["api_providers"]["sightengine"]["api_user_env"],
        config["detection"]["api_providers"]["sightengine"]["api_secret_env"],
        config["report"]["api_key_env"],
    ]
    assert all(re.fullmatch(r"[A-Z][A-Z0-9_]*", value) for value in values), "Use environment names, never credential literals"


def test_base_requirements_include_existing_entrypoint_dependencies():
    names = {
        re.split(r"[<>=!~\[]", line.strip(), maxsplit=1)[0]
        for line in (ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert {"datasets", "accelerate", "matplotlib", "seaborn", "pyarrow", "tqdm"} <= names


def test_model_catalog_is_valid_nonempty_json():
    assert json.loads((ROOT / "zhizengzeng-models.json").read_text())
