import numpy as np
import pytest
import torch

from research.image_attribution.openclip_route import (
    normalized, cache_identity, read_cache, write_cache, prototypes,
    knn_probabilities, rejection_report, fit_rejection_threshold, job_parts,
)


def row(i, source="a", split="train", role=None):
    return dict(sample_id=str(i), group_id=str(i), sha256=str(i), pixel_sha256=str(i),
                relative_path=f"images/{i}.png", source_label=source, split=split,
                is_ai=1, semantic_class="FFHQ" if role == "face" else "imagenet")


@pytest.mark.parametrize("value", [torch.zeros(1, 3), torch.tensor([[float("nan"), 1, 2]]),
                                   torch.tensor([[float("inf"), 1, 2]])])
def test_invalid_vectors_rejected(value):
    with pytest.raises(ValueError):
        normalized(value)


def test_cache_binds_ids_images_and_assets(tmp_path):
    rows = [row(1), row(2)]
    identity = cache_identity(rows, {"weights": "abc"}, {"module": "def"})
    path = tmp_path / "features.pt"
    write_cache(path, torch.eye(2), identity)
    assert torch.equal(read_cache(path, identity, 2), torch.eye(2))
    altered = [dict(rows[0], sha256="changed"), rows[1]]
    with pytest.raises(ValueError):
        read_cache(path, cache_identity(altered, {"weights": "abc"}, {"module": "def"}), 2)
    with pytest.raises(ValueError):
        read_cache(path, cache_identity(rows, {"weights": "different"}, {"module": "def"}), 2)


def test_holdout_never_cached():
    with pytest.raises(ValueError):
        cache_identity([row(1, split="holdout")], {}, {})


def test_train_prototypes_reject_unknowns():
    with pytest.raises(ValueError):
        prototypes(torch.eye(2), [row(1), row(2, "grok3")], ["a", "b"])
    assert torch.equal(prototypes(torch.eye(2), [row(1), row(2, "b")], ["a", "b"]), torch.eye(2))


def test_knn_uses_reference_only_and_ties_are_stable():
    p = knn_probabilities(torch.eye(2), torch.eye(2), torch.tensor([0, 1]), 2, 1)
    assert np.array_equal(p, np.eye(2))


def test_unknown_threshold_uses_calibration_only():
    rows = [row(1), row(2, "kling"), row(3, "kling")]
    threshold = fit_rejection_threshold(rows, ["a"], np.array([.99, .8, .9]), .05)
    assert threshold > .9
    for eval_score in [.1, .99]:
        result = rejection_report([row(4), row(5, "grok3")], ["a"],
                                  np.array([.95, eval_score]), np.array([0, 0]), threshold)
        assert result["acceptance_threshold"] == threshold
        assert result["known_retention"] == 1
        assert result["known_correct_and_retained"] == 1
        assert len(result["known_retention_wilson95"]) == 2


def test_open_split_excludes_unknown_train_and_selection():
    rows = [row(i, s, "train" if i < 10 else "dev")
            for i in range(30) for s in []]
    rows = [row(i, ["a", "kling", "grok3"][i % 3], "train" if i < 30 else "dev")
            for i in range(150)]
    rows.append(row(200, "a", "holdout"))
    parts, classes = job_parts(rows, dict(task="attribution", mode="open_set",
        unknown_calibration=["kling"], unknown_evaluation=["grok3"]), 42)
    assert classes == ["a"]
    for role in ("train", "selection"):
        assert all(r["source_label"] == "a" for r in parts[role])
    assert all(r["split"] != "holdout" for values in parts.values() for r in values)
    assert not any(r["source_label"] == "grok3" for r in parts["calibration"])
    assert not any(r["source_label"] == "kling" for r in parts["evaluation"])

def test_gpu_gate_targets_physical_zero_and_memory(monkeypatch):
    from scripts.run_image_openclip import gpu_idle
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    def busy(command, **kwargs):
        assert "--id=0" in command
        assert "--query-gpu=memory.used" in command
        return "1025"
    monkeypatch.setattr("subprocess.check_output", busy)
    with pytest.raises(RuntimeError, match="busy"):
        gpu_idle("cuda")
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: "1024")
    gpu_idle("cuda:0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="reserved"):
        gpu_idle("cuda")


def test_prediction_schema_has_explicit_role(tmp_path):
    import json
    from scripts.run_image_openclip import score_jsonl
    path = tmp_path / "calibration.jsonl"
    score_jsonl(path, [row(1)], ["a", "b"], np.array([[.8, .2]]))
    prediction = json.loads(path.read_text())
    assert prediction["role"] == "calibration"
    assert prediction["condition"] == "clean"
    assert prediction["sample_id"] == "1"
    assert prediction["classes"] == ["a", "b"]


def test_actual_image_cache_disagreement_blocks_completion(tmp_path, monkeypatch):
    import hashlib
    from PIL import Image
    from scripts.run_image_openclip import verify_images
    from research.image_attribution.openclip_route import encode_rows
    class Encoder(torch.nn.Module):
        def forward(self, x):
            return torch.cat([x.mean((2, 3))] * 256, dim=1)
    model = Encoder()
    transform = lambda image: torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1).copy())
    rows = []
    for i, color in enumerate([(20, 80, 180), (180, 80, 20)]):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (4, 4), color).save(path)
        rows.append(dict(row(i, split="dev"), relative_path=path.name,
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    monkeypatch.setattr("scripts.run_image_openclip.load_backbone", lambda *args: (model, transform))
    expected = encode_rows(model, transform, tmp_path, rows, "cpu")
    scores = lambda x: x[:, :2].softmax(1).numpy()
    result = verify_images({}, tmp_path, rows, expected, scores, "cpu")
    assert result["images_opened"] == 2
    with pytest.raises(ValueError, match="reproducibility"):
        verify_images({}, tmp_path, rows, expected.flip(0), scores, "cpu")


@pytest.mark.parametrize("task,mode", [("detection", "clean"), ("attribution", "open_set")])
def test_cpu_job_artifacts_and_30_epoch_protocol(tmp_path, monkeypatch, task, mode):
    import json
    from scripts import run_image_openclip as runner
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "verify_images", lambda *args: {"images_opened": 2})
    torch.set_num_threads(1)
    rows = []
    sources = ["a", "b", "kling", "grok3"] if mode == "open_set" else ["a", "b"]
    for i in range(180):
        r = row(i, sources[i % len(sources)], "train" if i < 40 else "dev")
        if task == "detection" and i % 2:
            r.update(is_ai=0, source_label=None)
        rows.append(r)
    experiment = dict(task=task, mode=mode)
    if mode == "open_set":
        experiment.update(unknown_calibration=["kling"], unknown_evaluation=["grok3"],
                          unknown_false_acceptance_target=.05)
    config = dict(seed=42, experiments={"test": experiment}, asset={}, dependencies={},
                  manifest_sha256="manifest", selection_sha256="selection", implementation_sha256={},
                  limitations=[], bootstrap_repeats=2,
                  training=dict(epochs=30, batch_size=32, learning_rate=.001,
                                scheduler_step=1, scheduler_gamma=.9))
    features = normalized(torch.randn(len(rows), 768, generator=torch.Generator().manual_seed(42)))
    output = tmp_path / "job"
    output.mkdir()
    runner.run_job(config, "test", rows, features, rows, tmp_path, output, "cpu")
    assert json.loads((output / "status.json").read_text())["status"] == "completed"
    history = [json.loads(line) for line in (output / "linear/history.jsonl").read_text().splitlines()]
    assert [item["epoch"] for item in history] == list(range(1, 31))
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert {p["role"] for p in predictions} == {"calibration", "evaluation"}
    if mode == "open_set":
        metrics = json.loads((output / "metrics.json").read_text())
        assert set(metrics["rejection"]) == {"msp", "prototype_cosine"}
        assert metrics["classes"] == ["a", "b"]
        for report in metrics["rejection"].values():
            assert report["calibration"]["unknown_false_acceptance"] <= .05


def test_disk_gate_requires_twenty_gib(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from scripts.run_image_openclip import require_disk_space
    monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=20 * 1024**3 - 1))
    with pytest.raises(RuntimeError, match="20 GiB"):
        require_disk_space(tmp_path)
    monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=20 * 1024**3))
    require_disk_space(tmp_path)


def test_low_disk_refuses_before_output_or_backbone(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import run_image_openclip as runner
    (tmp_path / "outputs/image_experiments").mkdir(parents=True)
    (tmp_path / "config.json").write_text(json.dumps(
        dict(execution_enabled=True, output_root="outputs/image_experiments/new-openclip")))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "prepare", lambda config: (tmp_path, [], [], {}, {}))
    monkeypatch.setattr("sys.argv", ["run_image_openclip.py", "--config", "config.json", "--execute", "--device", "cpu"])
    monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=0))
    monkeypatch.setattr(runner, "load_backbone", lambda *args: pytest.fail("backbone must not load"))
    with pytest.raises(RuntimeError, match="20 GiB"):
        runner.main()
    assert not (tmp_path / "outputs/image_experiments/new-openclip").exists()
