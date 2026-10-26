import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research.image_attribution.followups import (
    partition_rows, choose_threshold, binary_report, open_set_report,
    family_scores, Postprocess, validate_probabilities,
)


def rows():
    result = []
    for split, n in [('train', 12), ('dev', 90), ('holdout', 6)]:
        for i in range(n):
            for source in ['a', 'b', 'u_cal', 'u_eval', None]:
                result.append(dict(sample_id=f'{split}-{i}-{source}', group_id=f'{split}-{i}',
                    split=split, source_label=source, is_ai=int(source is not None),
                    semantic_class='cat' if i % 2 else 'dog'))
    return result


def spec(**kwargs):
    return dict(task='attribution', mode='open_set', seed=42,
        unknown_calibration=['u_cal'], unknown_evaluation=['u_eval'], **kwargs)


def test_unknown_sources_never_train_and_holdout_never_enters_roles():
    parts = partition_rows(rows(), spec())
    assert {r['source_label'] for r in parts['train']} == {'a', 'b'}
    assert {r['source_label'] for r in parts['selection']} == {'a', 'b'}
    assert {r['source_label'] for r in parts['calibration']} == {'a', 'b', 'u_cal'}
    assert {r['source_label'] for r in parts['evaluation']} == {'a', 'b', 'u_eval'}
    groups = [{r['group_id'] for r in p} for p in parts.values()]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i+1:])
    assert all(r['split'] != 'holdout' for p in parts.values() for r in p)
    assert partition_rows(list(reversed(rows())), spec()) == parts


def test_held_semantic_is_absent_from_training_and_selection():
    parts = partition_rows(rows(), dict(task='detection', mode='cross_semantic', seed=42,
                                       held_semantics=['cat']))
    assert {r['semantic_class'] for r in parts['train']} == {'dog'}
    assert {r['semantic_class'] for r in parts['selection']} == {'dog'}
    assert {r['semantic_class'] for r in parts['evaluation']} == {'cat'}


def test_source_holdout_retains_real_controls_but_no_held_ai_training():
    parts = partition_rows(rows(), dict(task='detection', mode='cross_source', seed=42,
                                       held_sources=['a']))
    assert 'a' not in {r['source_label'] for r in parts['train']}
    assert {r['source_label'] for r in parts['evaluation']} == {'a', None}


def test_invalid_unknown_overlap_and_leaking_groups_fail_closed():
    with pytest.raises(ValueError):
        partition_rows(rows(), dict(spec(), unknown_evaluation=['u_cal']))
    bad = rows(); bad[-1]['group_id'] = 'train-0'
    with pytest.raises(ValueError):
        partition_rows(bad, spec())


def test_threshold_ties_cannot_violate_fpr_budget():
    t = choose_threshold([.1, .2, .2, .9], .25)
    assert t > .2 and t <= .9
    assert np.mean(np.array([.1, .2, .2, .9]) >= t) == .25
    assert choose_threshold([1., 1.], 0) > 1.
    report = binary_report([0, 0, 1, 1], [.1, .9, .8, .99], t)
    assert report['fpr'] == .5
    assert report['tpr'] == 1.
    assert report['negative_count'] == 2
    assert report['fpr_resolution'] == .5
    assert 0 <= report['fpr_wilson95'][0] < .5 < report['fpr_wilson95'][1] <= 1


def test_open_set_scores_use_unknown_positive_and_count_rejected_known_errors():
    report = open_set_report(['a', 'a', 'x', 'x'], ['a', 'b'],
                             np.array([[.9,.1], [.55,.45], [.5,.5], [.51,.49]]), .8)
    assert report['unknown_auroc'] == 1.
    assert report['known_acceptance'] == .5
    assert report['known_correct_and_accepted_rate'] == .5
    assert report['unknown_rejection_rate'] == 1.


def test_family_aggregation_sums_probabilities_not_argmax_labels():
    mapping = {'a': {'family': 'F', 'evidence': 'publisher-label'},
               'b': {'family': 'F', 'evidence': 'publisher-label'},
               'c': {'family': 'G', 'evidence': 'publisher-label'}}
    names, scores = family_scores(['a','b','c'], np.array([[.3,.3,.4]]), mapping)
    assert names == ['F','G']
    np.testing.assert_allclose(scores, [[.6,.4]])
    with pytest.raises(ValueError):
        family_scores(['a','missing'], np.array([[.5,.5]]), mapping)
    with pytest.raises(ValueError):
        family_scores(['a'], np.array([[1.]]), {'a': {'family':'F','evidence':None}})


def test_probabilities_reject_nan_wrong_width_or_nonprobability_scores():
    for p in [np.array([[np.nan,.5]]), np.array([[.1,.1]]), np.array([[1.]]), np.array([[-.1,1.1]])]:
        with pytest.raises(ValueError): validate_probabilities(p, 1, 2)


def test_postprocess_preserves_input_and_returns_rgb():
    image = Image.new('RGB', (300,250), color=(17,22,39))
    original = image.tobytes()
    changed = Postprocess(jpeg_quality=75, scale=.5)(image)
    assert image.tobytes() == original
    assert changed.mode == 'RGB' and changed.size == image.size
    assert changed is not image


def test_default_cli_only_plans_and_execution_requires_explicit_flag(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / 'scripts/run_image_followup.py'
    cfg = root / 'configs/image_experiments/iab_followups_v4_20.json'
    result = subprocess.run([sys.executable, str(script), '--config', str(cfg), '--experiment', 'D1'],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['execution_enabled'] is False
    assert list(tmp_path.iterdir()) == []
    result = subprocess.run([sys.executable, str(script), '--config', str(cfg), '--experiment', 'D1',
                             '--action', 'train'], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode != 0
    assert list(tmp_path.iterdir()) == []


def test_augmentation_changes_training_only_not_dev_or_train_metric_inputs(tmp_path):
    from scripts.run_image_experiment import image_datasets
    from torchvision import transforms
    Image.new('RGB',(300,250),'white').save(tmp_path/'sample.png')
    row = {'relative_path':'sample.png'}
    parts = {'train':([row],[0]), 'dev':([row],[0])}
    datasets, clean_train = image_datasets(parts,tmp_path,
        transforms.Compose([transforms.Lambda(lambda im: Image.new('RGB',im.size,'black')),
                            transforms.ToTensor()]))
    assert float(datasets['train'][0][0].max()) == 0.
    assert float(datasets['dev'][0][0].min()) > 2.
    assert float(clean_train[0][0].min()) > 2.
    default, default_train = image_datasets(parts,tmp_path)
    assert default['train'] is default_train


def test_d3_evaluation_labels_never_change_fitted_thresholds():
    from scripts.run_image_followup import evaluate_scores
    cal = [dict(is_ai=y,group_id=str(i)) for i,y in enumerate([0,0,1,1])]
    query = [dict(is_ai=y,source_label='a' if y else None,semantic_class='cat') for y in [0,0,1,1]]
    experiment = dict(task='detection',mode='calibration',seed=42,target_fprs=[0.],bootstrap_repeats=10)
    scores = dict(calibration=np.array([[.9,.1],[.8,.2],[.1,.9],[.2,.8]]),
                  evaluation=np.array([[.9,.1],[.8,.2],[.1,.9],[.2,.8]]))
    first = evaluate_scores(experiment,dict(calibration=cal,evaluation=query),['real','ai'],scores)
    scores['evaluation'] = scores['evaluation'][::-1]
    second = evaluate_scores(experiment,dict(calibration=cal,evaluation=query),['real','ai'],scores)
    assert first['calibrated']['0.0']['evaluation']['threshold'] == second['calibrated']['0.0']['evaluation']['threshold']
    assert first['fixed_0_5']['accuracy'] == 1.
    assert second['fixed_0_5']['accuracy'] == 0.


def test_class_coverage_guard_does_not_silently_run_partial_unknown_protocol():
    from scripts.run_image_followup import validate_parts
    parts = partition_rows(rows(),spec())
    validate_parts(parts,spec())
    parts['evaluation'] = [r for r in parts['evaluation'] if r['source_label']!='u_eval']
    with pytest.raises(ValueError): validate_parts(parts,spec())


def test_pixel_duplicates_cannot_cross_development_roles():
    fixture = rows()
    # Same decoded image under different group IDs must still not leak.
    for row in fixture:
        if row['split']=='dev': row['pixel_sha256']='same-pixels'
    with pytest.raises(ValueError): partition_rows(fixture,spec())


def test_one_epoch_synthetic_training_saves_reloads_and_reports(tmp_path,monkeypatch):
    import torch
    from types import SimpleNamespace
    from scripts import run_image_experiment as runner
    from research.image_attribution.followups import TrainingAugmentation
    from torchvision import transforms

    class TinyClassifier(torch.nn.Module):
        def __init__(self,n,pretrained_path=None):
            super().__init__()
            self.pool=torch.nn.AdaptiveAvgPool2d(1)
            self.fc=torch.nn.Linear(3,n)
        def forward(self,x): return self.fc(self.pool(x).flatten(1))

    def reload_tiny(path,device):
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        model=TinyClassifier(len(checkpoint['classes']))
        model.load_state_dict(checkpoint['state_dict'])
        return model.to(device).eval(),checkpoint

    monkeypatch.setattr(runner,'ResNetClassifier',TinyClassifier)
    monkeypatch.setattr(runner,'load_checkpoint',reload_tiny)
    torch.set_num_threads(1)
    fixture=[]
    for split in ['train','dev']:
        for i in range(4):
            path=tmp_path/f'{split}-{i}.png'
            Image.new('RGB',(260,260),'white' if i%2 else 'black').save(path)
            fixture.append(dict(sample_id=path.stem,relative_path=path.name,split=split,
                is_ai=i%2,source_label='a' if i%2 else None,semantic_class='cat'))
    manifest=tmp_path/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in fixture))
    weights=tmp_path/'pretrained-fixture'; weights.write_text('synthetic fixture; no real model')
    out=tmp_path/'run';out.mkdir()
    args=SimpleNamespace(task='detection',manifest=manifest,weights=weights,device='cpu',
        batch_size=2,workers=0,epochs=1,
        train_transform=transforms.Compose([TrainingAugmentation(),runner.resnet_transform()]))
    result=runner.train(args,fixture,out)
    assert result['train_counts']=={'real':2,'ai':2}
    assert result['dev_counts']=={'real':2,'ai':2}
    assert result['selected_epoch']==1
    assert result['reload_max_abs_error'] < 1e-5
    assert len((out/'predictions.jsonl').read_text().splitlines())==4
    assert json.loads((out/'history.json').read_text())[0]['train']['class_counts']=={'real':2,'ai':2}
