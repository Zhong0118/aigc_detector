"""Offline CPU contracts; no official model weights or image datasets required."""
import hashlib
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from scripts import evaluate_omniaid as omni


class TinyDetector(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.probability = torch.nn.Parameter(torch.tensor(.8))
        self.config = config

    def forward(self, images):
        return self.probability.expand(len(images))


@pytest.fixture
def assets(tmp_path):
    paths, config = {}, {'assets': {}}
    for name in omni.ASSET_NAMES:
        filename = {'clip_config': 'clip/config.json', 'clip_weights': 'clip/model.safetensors'}.get(name, name)
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
        paths[name] = path
    paths['model_config'].write_text(json.dumps(omni.MODEL_PARAMETERS))
    torch.save({'model': {'module.probability': torch.tensor(.8)}}, paths['checkpoint'])
    for name, path in paths.items():
        config['assets'][name] = dict(path=str(path), bytes=path.stat().st_size,
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return paths, config


def test_disabled_run_never_loads_model_or_manifest(tmp_path, monkeypatch):
    loader = Mock(side_effect=AssertionError('must not load'))
    monkeypatch.setattr(omni, 'load_detector', loader)
    with pytest.raises(PermissionError):
        omni.run(tmp_path, {'execution_enabled': False}, 'cpu')
    loader.assert_not_called()


def test_missing_weights_fail_before_author_code_import(tmp_path, assets, monkeypatch):
    paths, config = assets
    paths['clip_weights'].unlink()
    source = Mock(side_effect=AssertionError('must not import'))
    monkeypatch.setattr(omni, 'reviewed_source', source)
    with pytest.raises(FileNotFoundError):
        omni.load_detector(tmp_path, config, 'cpu')
    source.assert_not_called()


def test_unregistered_converted_weights_are_rejected(tmp_path, assets):
    _, config = assets
    config['assets']['clip_weights']['sha256'] = None
    with pytest.raises(ValueError, match='unbound'):
        omni.verify_assets(tmp_path, config)


def test_unknown_source_is_never_executed(tmp_path):
    source = tmp_path / 'unknown.py'
    source.write_text('raise RuntimeError("this code must never run")')
    with pytest.raises(ValueError, match='reviewed'):
        omni.reviewed_source(source)


def test_loader_uses_exact_v2_parameters_and_freezes_checkpoint(tmp_path, assets, monkeypatch):
    _, config = assets
    monkeypatch.setattr(omni, 'reviewed_source', lambda path: SimpleNamespace(OmniAID=TinyDetector))
    model, _ = omni.load_detector(tmp_path, config, 'cpu')
    assert model.config.num_experts == 6 and model.config.rank_per_expert == 1
    assert model.config.moe_top_k == 2 and model.config.moe_router_hidden_dim == 256
    assert model.config.is_hybrid is True
    assert not model.training and not next(model.parameters()).requires_grad


def test_checkpoint_key_mismatch_is_not_silently_ignored(tmp_path, assets, monkeypatch):
    paths, config = assets
    torch.save({'model': {'wrong_key': torch.tensor(.8)}}, paths['checkpoint'])
    config['assets']['checkpoint'].update(bytes=paths['checkpoint'].stat().st_size,
                                        sha256=hashlib.sha256(paths['checkpoint'].read_bytes()).hexdigest())
    monkeypatch.setattr(omni, 'reviewed_source', lambda path: SimpleNamespace(OmniAID=TinyDetector))
    with pytest.raises(RuntimeError):
        omni.load_detector(tmp_path, config, 'cpu')


def test_author_probability_is_not_sigmoided_again():
    loader = [(torch.zeros(2, 3, 336, 336), torch.tensor([0, 1]))]
    scores = omni.score_loader(TinyDetector().eval(), loader, 'cpu')
    np.testing.assert_allclose(scores, [[.2, .8], [.2, .8]], atol=1e-6)


def test_out_of_range_author_output_fails():
    model = TinyDetector()
    model.probability.data.fill_(1.2)
    with pytest.raises(ValueError, match='probability'):
        omni.score_loader(model, [(torch.zeros(1, 3, 336, 336), torch.tensor([0]))], 'cpu')


def test_unbound_manifest_is_rejected_before_reading_images(tmp_path):
    with pytest.raises(ValueError, match='manifest'):
        omni.load_parts(tmp_path, {'dataset': {'manifest_sha256': None}}, 42)


def bound_manifest(tmp_path, monkeypatch):
    rows = [dict(sample_id=str(i), group_id=str(i), split='dev', is_ai=bool(i % 2),
                 source_label='generator' if i % 2 else None, semantic_class='fixture') for i in range(120)]
    rows += [dict(rows[0], sample_id='holdout', group_id='holdout', split='holdout')]
    for row in rows:
        row['relative_path'] = row['sample_id'] + '.png'
        row['sha256'] = hashlib.sha256(row['sample_id'].encode()).hexdigest()
    selected = omni.partition_rows(rows, dict(task='detection', mode='calibration', seed=42))
    # No selection/holdout image files exist, so reading one is a test failure.
    for role in ('calibration', 'evaluation'):
        for row in selected[role]:
            (tmp_path / row['relative_path']).write_bytes(row['sample_id'].encode())
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    selection = tmp_path / 'selection.json'
    selection.write_text('{}')
    config = {'dataset': dict(path=str(tmp_path), manifest_sha256=omni.sha256(manifest),
                              selection_sha256=omni.sha256(selection))}
    monkeypatch.setattr(omni, 'validate_manifest', lambda *args, **kwargs: None)
    return config


def test_only_calibration_and_evaluation_files_are_hashed(tmp_path, monkeypatch):
    config = bound_manifest(tmp_path, monkeypatch)
    _, parts = omni.load_parts(tmp_path, config, 42)
    assert set(parts) == {'calibration', 'evaluation'}
    assert all(row['split'] == 'dev' for values in parts.values() for row in values)
    assert all(row['sample_id'] != 'holdout' for values in parts.values() for row in values)


def test_modified_evaluation_file_is_rejected(tmp_path, monkeypatch):
    config = bound_manifest(tmp_path, monkeypatch)
    _, parts = omni.load_parts(tmp_path, config, 42)
    (tmp_path / parts['evaluation'][0]['relative_path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='image hash'):
        omni.load_parts(tmp_path, config, 42)


def test_cli_execution_defaults_to_cpu(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    path.write_text('{}')
    run = Mock(return_value={})
    monkeypatch.setattr(omni, 'run', run)
    monkeypatch.setattr(omni.sys, 'argv', ['evaluate_omniaid.py', '--config', str(path), '--execute'])
    omni.main()
    assert run.call_args.args[2] == 'cpu'


def test_explicit_cuda_is_isolated_to_the_configured_physical_gpu(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7')
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    assert omni.resolve_device({'gpu_index': 1}, 'cuda:7') == 'cuda:0'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'


def ready_config():
    return dict(execution_enabled=True, status='ready', automatic_execution=False,
                task='detection', epoch=None, gpu_index=1, output='evaluation-output')


def test_verify_and_execute_refuse_a_busy_shared_research_lock(tmp_path, monkeypatch):
    loader = Mock(side_effect=AssertionError('busy means no model loading'))
    monkeypatch.setattr(omni, 'load_detector', loader)
    with omni.research_lock(tmp_path):
        with pytest.raises(RuntimeError, match='busy'):
            omni.verify_load(tmp_path, {}, 'cpu')
        with pytest.raises(RuntimeError, match='busy'):
            omni.run(tmp_path, ready_config(), 'cpu')
    loader.assert_not_called()
    assert not (tmp_path / 'evaluation-output').exists()


def test_failed_run_records_status_without_overwriting_an_existing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(omni, 'verify_assets', Mock(side_effect=ValueError('fixture asset failure')))
    config = ready_config()
    with pytest.raises(ValueError, match='fixture asset failure'):
        omni.run(tmp_path, config, 'cpu')
    out = tmp_path / config['output']
    previous = (out / 'status.json').read_bytes()
    status = json.loads(previous)
    assert status['status'] == 'failed' and status['exit_code'] == 1
    assert json.loads((out / 'resolved_config.json').read_text())['runtime_device'] == 'cpu'
    with pytest.raises(FileExistsError):
        omni.run(tmp_path, config, 'cpu')
    assert (out / 'status.json').read_bytes() == previous


def test_success_records_running_then_completed(tmp_path, monkeypatch):
    def evaluate(root, config, device, out):
        assert json.loads((out / 'status.json').read_text())['status'] == 'running'
        assert (out / 'resolved_config.json').is_file()
        return {'fixture': True}
    monkeypatch.setattr(omni, '_evaluate', evaluate)
    result = omni.run(tmp_path, ready_config(), 'cpu')
    assert result == {'fixture': True}
    status = json.loads((tmp_path / 'evaluation-output/status.json').read_text())
    assert status['status'] == 'completed' and status['exit_code'] == 0
