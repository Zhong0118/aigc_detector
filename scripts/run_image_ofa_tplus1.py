#!/usr/bin/env python3
"""One-shot frozen OFA simulator + 20 known / one rejection head project adaptation."""
import argparse
import fcntl
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from research.image_attribution.models import load_ofa, ofa_features, ofa_transform
from research.image_attribution.ofa_tplus1 import (
    DIMENSION, RECIPE, UNKNOWN, calibrate, fit_head, head_probabilities,
    reconstruct, report, safe_load, select_reconstructions,
)
from research.image_attribution.openclip_route import digest, image_tensor, job_parts, normalized, write_cache
from research.image_attribution.protocol import sha256, validate_manifest, write_json
from scripts.run_image_openclip import gpu_idle, require_disk_space

DEFAULT_CONFIG = 'configs/image_experiments/iab_ofa_tplus1_q179_20261029.json'
EXPERIMENT = dict(task='attribution', mode='open_set', unknown_calibration=['kling'], unknown_evaluation=['grok3'])
PREPROCESSING = dict(input_size=256, rgb=True, resize='short_edge_only_if_below_256',
    crop='center', normalization_mean=[.5]*3, normalization_std=[.5]*3,
    reconstruction_quantization='none', residual='clamp_minus1_plus1_then_01_dncnn_sigma1_div256',
    fingerprint='fft2_abs_log1p_no_fftshift_128_l2', denoiser_microbatch=1,
    simulator_microbatch=1, native_architecture_replay='isolated_python_random_seed_not_arch_json')


def checked_asset(spec):
    path = ROOT/spec['path']
    if path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
        raise ValueError(f'asset identity changed: {path}')
    return path


def feature_identity(rows, config):
    if any(r['split'] == 'holdout' for r in rows):
        raise ValueError('holdout forbidden in cache')
    ids = [r['sample_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate cache IDs')
    return dict(schema='ofa-tplus1-feature-cache-v1', ids=ids, rows_sha256=digest(rows),
        assets=config['assets'], code=config['implementation_sha256'],
        preprocessing=config['preprocessing'], dimensions=DIMENSION)


def prepare(config):
    if (config['training'] != RECIPE or config['experiment'] != EXPERIMENT
            or config['preprocessing'] != PREPROCESSING or config['unknown_training_count'] != 1024
            or config['unknown_false_acceptance_target'] != .05
            or config['architecture_seeds'] != [int(digest(['ofa-simulator',42,i])[:16],16) for i in range(8)]):
        raise ValueError('only registered frozen OFA 20+1 recipe allowed')
    data = ROOT/config['dataset']
    for name, key in [('manifest.jsonl','manifest_sha256'),('selection.json','selection_sha256')]:
        if sha256(data/name) != config[key]:
            raise ValueError('dataset identity changed')
    for name, expected in config['dependencies'].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f'dependency changed: {name}')
    for path, expected in config['implementation_sha256'].items():
        if sha256(ROOT/path) != expected:
            raise ValueError(f'implementation changed: {path}')
    for spec in config['assets'].values():
        checked_asset(spec)
    rows = [json.loads(line) for line in (data/'manifest.jsonl').read_text().splitlines()]
    validate_manifest(rows, data, verify_files=False)
    parts, classes = job_parts(rows, EXPERIMENT, 42)
    parts = {role:[dict(row,role=role) for row in values] for role,values in parts.items()}
    if classes != config['known_classes'] or len(classes) != 20:
        raise ValueError('known source vocabulary changed')
    if {k:len(v) for k,v in parts.items()} != config['expected_role_counts']:
        raise ValueError('real role counts changed')
    synthetic = select_reconstructions(rows, 42)
    # Native architecture labels must not become a proxy for a subset of semantic domains.
    semantics = {r['semantic_class'] for r in synthetic}
    if any({r['semantic_class'] for r in synthetic if r['architecture_index']==i} != semantics for i in range(8)):
        raise ValueError('each architecture must cover every training semantic')
    return data, parts, classes, synthetic


def load_fingerprint(config, device):
    for key in ('fingerprint','denoiser'):
        safe_load(checked_asset(config['assets'][key]))
    encoder, pair, denoiser = load_ofa(checked_asset(config['assets']['fingerprint']),
        checked_asset(config['assets']['denoiser']), device)
    del pair
    return encoder, denoiser


def load_simulator(config, device):
    modules = []
    for key, name in [('simulator_encoder_source','Encoder'),('simulator_decoder_source','Decoder')]:
        path = checked_asset(config['assets'][key])
        spec = importlib.util.spec_from_file_location('ofa_tplus1_' + path.stem, path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        modules.append(getattr(module,name)())
    checkpoint = safe_load(checked_asset(config['assets']['simulator']))
    for module,key in zip(modules,('encoder','decoder')):
        module.load_state_dict(checkpoint[key],strict=True)
        module.eval().requires_grad_(False).to(device)
    return modules


def encode_known(encoder, denoiser, data, rows, device, batch_size=8, progress=None):
    output = []
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            x = torch.stack([image_tensor(data,r,ofa_transform) for r in rows[start:start+batch_size]])
            output.append(normalized(ofa_features(x.to(device),encoder,denoiser),DIMENSION).cpu())
            if progress and (start % (batch_size*32)==0 or start+batch_size>=len(rows)):
                progress(min(start+batch_size,len(rows)),len(rows))
    return torch.cat(output)


def verify_selection(config,data,rows,cached,head,device):
    if len(rows)<2 or any(r['role']!='selection' or r['split']!='dev' for r in rows):
        raise ValueError('two actual selection images required')
    encoder,denoiser = load_fingerprint(config,device)
    batch = encode_known(encoder,denoiser,data,rows[:2],device,2)
    single = encode_known(encoder,denoiser,data,rows[:2],device,1)
    checks = dict(images_opened=2,selection_sample_ids=[r['sample_id'] for r in rows[:2]],
        full_offline_backbone_reload=True,
        cache_feature_max_error=float((batch-cached[:2]).abs().max()),
        single_batch_feature_max_error=float((batch-single).abs().max()),
        cache_score_max_error=float(np.abs(head_probabilities(head,batch,device)-head_probabilities(head,cached[:2],device)).max()),
        single_batch_score_max_error=float(np.abs(head_probabilities(head,batch,device)-head_probabilities(head,single,device)).max()))
    if any(value>1e-5 for key,value in checks.items() if key.endswith('max_error')):
        raise ValueError(f'actual selection reproducibility failed: {checks}')
    del encoder,denoiser
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    return checks


def prediction_record(row, classes, value, threshold):
    value = np.asarray(value,dtype=np.float64)
    item = {k:row[k] for k in ('sample_id','group_id','source_label','is_ai','semantic_class','role','split')}
    item.update(classes=classes+[UNKNOWN],scores=value.tolist(),condition='clean',
        known_prediction=classes[int(value[:-1].argmax())],
        simulated_unknown_score=float(value[-1]),acceptance_confidence=float(1-value[-1]),
        accepted=bool(1-value[-1]>=threshold))
    return item


def new_output(path):
    Path(path).mkdir(parents=True,exist_ok=False)


def save_cache(path,features,identity):
    write_cache(path,features,identity)
    saved = safe_load(path)
    if saved['identity'] != identity:
        raise ValueError('saved cache identity mismatch')
    result = normalized(saved['features'],DIMENSION)
    if result.shape != features.shape or float((result-features).abs().max())>1e-6:
        raise ValueError('saved feature cache differs from extracted features')
    return result


def run(config,data,parts,classes,synthetic,out,device,state):
    write_json(out/'roles.json',{k:[r['sample_id'] for r in v] for k,v in parts.items()})
    with (out/'derived_training.jsonl').open('x') as stream:
        for row in synthetic: stream.write(json.dumps(row,allow_nan=False)+'\n')
    encoder,denoiser = load_fingerprint(config,device)
    rows = sorted([r for values in parts.values() for r in values],key=lambda r:r['sample_id'])
    identity = feature_identity(rows,config)
    state(status='running',stage='known_features',completed_images=0,total_images=len(rows))
    features = encode_known(encoder,denoiser,data,rows,device,8,
        progress=lambda n,t:state(status='running',stage='known_features',completed_images=n,total_images=t))
    features = save_cache(out/'known_features.pt',features,identity)
    simulator_encoder,simulator_decoder = load_simulator(config,device)
    traces,unknown_features = {},[]
    state(status='running',stage='simulator_features',completed_images=0,total_images=1024)
    with torch.inference_mode():
        for i,row in enumerate(synthetic):
            architecture = row['architecture_index']
            x = image_tensor(data,row['parent'],ofa_transform).unsqueeze(0).to(device)
            reconstruction,trace = reconstruct(simulator_encoder,simulator_decoder,x,
                config['architecture_seeds'][architecture],traces.get(architecture))
            traces[architecture] = trace
            unknown_features.append(normalized(ofa_features(reconstruction,encoder,denoiser),DIMENSION).cpu())
            if (i+1)%32==0:
                state(status='running',stage='simulator_features',completed_images=i+1,total_images=1024)
    if len({digest(t) for t in traces.values()}) != 8:
        raise ValueError('registered architecture seeds did not yield eight distinct native traces')
    trace_receipt = {str(i):dict(seed=config['architecture_seeds'][i],trace=t,sha256=digest(t)) for i,t in traces.items()}
    write_json(out/'architecture_traces.json',trace_receipt)
    synthetic_identity = dict(feature_identity(synthetic,config),architecture_traces=trace_receipt)
    unknown_features = save_cache(out/'unknown_train_features.pt',torch.cat(unknown_features),synthetic_identity)
    del encoder,denoiser,simulator_encoder,simulator_decoder
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    index = {row['sample_id']:i for i,row in enumerate(rows)}
    x = {role:features[[index[r['sample_id']] for r in selected]] for role,selected in parts.items()}
    train_x = torch.cat([x['train'],unknown_features])
    train_y = torch.tensor([classes.index(r['source_label']) for r in parts['train']] + [20]*1024,dtype=torch.long)
    state(status='running',stage='head_training')
    head,metadata = fit_head(train_x,train_y,x['selection'],parts['selection'],classes,out,device)
    saved = safe_load(out/'head.pth')
    if saved['classes'] != classes+[UNKNOWN] or saved['input_dimensions'] != DIMENSION:
        raise ValueError('saved head class or feature order changed')
    reloaded = torch.nn.Linear(DIMENSION,21).to(device)
    reloaded.load_state_dict(saved['state_dict'],strict=True);reloaded.eval()
    head_error = float(np.abs(head_probabilities(head,x['selection'][:8],device)-head_probabilities(reloaded,x['selection'][:8],device)).max())
    if head_error>1e-6: raise ValueError('saved head reload mismatch')
    state(status='running',stage='actual_selection_reload')
    checks = verify_selection(config,data,parts['selection'],x['selection'],reloaded,device)
    checks['head_reload_max_error'] = head_error
    write_json(out/'actual_selection_verification.json',checks)
    # No calibration/evaluation probabilities are computed before the actual-image reload gate.
    state(status='running',stage='calibration_and_evaluation')
    scores = {role:head_probabilities(reloaded,x[role],device) for role in ('calibration','evaluation')}
    threshold = calibrate(parts['calibration'],classes,scores['calibration'])
    results = {role:report(parts[role],classes,scores[role],threshold) for role in scores}
    with (out/'predictions.jsonl').open('x') as stream:
        for role,p in scores.items():
            for row,value in zip(parts[role],p):
                item = prediction_record(row,classes,value,threshold)
                stream.write(json.dumps(item,allow_nan=False)+'\n')
    write_json(out/'metrics.json',dict(metadata,rejection=dict(score_method='one_minus_simulated_unknown',
        acceptance_threshold=threshold,threshold_fitted_role='calibration',unknown_false_acceptance_target=.05,
        calibration=results['calibration'],evaluation=results['evaluation']),actual_selection_verification=checks,
        holdout_scored=False,synthetic_unknown_training_n=1024,limitations=config['limitations']))
    write_json(out/'provenance.json',dict(config_sha256=digest(config),config=config,
        role_ids_sha256={k:digest([r['sample_id'] for r in v]) for k,v in parts.items()},
        known_cache_sha256=sha256(out/'known_features.pt'),unknown_cache_sha256=sha256(out/'unknown_train_features.pt'),
        head_sha256=sha256(out/'head.pth'),architecture_traces_sha256=sha256(out/'architecture_traces.json'),
        synthetic_manifest_sha256=sha256(out/'derived_training.jsonl'),
        holdout_read_for_inference=False,actual_selection_verification=checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=DEFAULT_CONFIG)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--device',default='cuda',choices=['cuda','cuda:0'])
    args = parser.parse_args()
    config = json.loads((ROOT/args.config).read_text())
    data,parts,classes,synthetic = prepare(config)
    if not args.execute:
        print(json.dumps(dict(status='prepared_disabled' if not config['execution_enabled'] else 'prepared',
            role_counts={k:len(v) for k,v in parts.items()},known_classes=classes,
            synthetic_train_n=len(synthetic),synthetic_manifest_sha256=digest(synthetic),
            images_opened=False,holdout_scored=False),indent=2));return
    if config.get('execution_enabled') is not True:
        raise PermissionError('registered execution is disabled pending review')
    with (ROOT/'outputs/image_experiments/gpu0-research.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu_idle(args.device);require_disk_space(ROOT)
        out = ROOT/config['output_root'];new_output(out)
        launch = dict(pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            process_start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            command_line=Path('/proc/self/cmdline').read_bytes().decode().split(chr(0))[:-1],started_unix=time.time())
        write_json(out/'launch.json',launch);write_json(out/'registration.json',config)
        def state(**kwargs):write_json(out/'status.json',dict(launch=launch,updated_unix=time.time(),**kwargs))
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        try:
            run(config,data,parts,classes,synthetic,out,args.device,state)
            state(status='completed',exit_code=0)
        except Exception as error:
            state(status='failed',exit_code=1,error=repr(error));raise

if __name__=='__main__': main()
