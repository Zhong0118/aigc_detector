#!/usr/bin/env python3
"""One-shot idle waiter for the registered ConvNeXt run; never retries a child."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'outputs/image-remaining-routes-20261029/queue-convnext-v1'
CONFIG=ROOT/'configs/image_experiments/iab_convnext_q179_20261029.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def identity(pid):
    p=Path('/proc')/str(pid)
    return dict(pid=pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        process_start_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19])


def alive(record):
    try:
        p=Path('/proc')/str(record['pid'])
        return identity(record['pid'])=={k:record[k] for k in identity(record['pid'])} and p.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except (FileNotFoundError,ProcessLookupError):return False


def check_registration(binding):
    for path,digest in binding.items():
        if sha(ROOT/path)!=digest:raise ValueError(f'registered input changed: {path}')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:raise SystemExit('explicit --execute required')
    state_parent=STATE.parent
    with (state_parent/'convnext-waiter.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=json.loads((state_parent/'convnext-busy-preflight.json').read_text())
        if previous['status']!='blocked_before_model_execution' or alive(previous['old_launch']):
            raise ValueError('prior preflight attempt is not confirmed dead and blocked')
        cfg=json.loads(CONFIG.read_text());output=ROOT/cfg['output_root']
        if output.exists():raise FileExistsError('preserve any existing model output')
        STATE.mkdir(exist_ok=False)
        launch=identity(os.getpid());launch['started_unix']=time.time()
        binding={str(CONFIG.relative_to(ROOT)):sha(CONFIG),str(Path(__file__).relative_to(ROOT)):sha(__file__),**cfg['implementation_sha256']}
        write(STATE/'registration.json',dict(inputs=binding,config_sha256=sha(CONFIG),previous_preflight=previous))
        write(STATE/'launch.json',launch)
        def save(**kwargs):write(STATE/'status.json',dict(launch=launch,updated_unix=time.time(),**kwargs))
        try:
            while True:
                check_registration(binding)
                if output.exists():raise FileExistsError('another process created model output while waiting')
                if shutil.disk_usage(ROOT).free<20*1024**3:raise RuntimeError('disk below20GiB')
                used=int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                available=False
                with (ROOT/'outputs/image_experiments/gpu0-research.lock').open('a') as research_lock:
                    try:
                        fcntl.flock(research_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);available=True
                    except BlockingIOError:pass
                if used<=1024 and available:break
                save(status='waiting_gpu0',used_mib=used,research_lock_available=available)
                time.sleep(60)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
            command=[str(ROOT/'.venv-image-next/bin/python'),'scripts/run_image_convnext.py','--execute']
            with (STATE/'worker.log').open('xb') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                child_identity=identity(child.pid);write(STATE/'child-launch.json',dict(child_identity,command=command))
                save(status='running',child=child_identity)
                code=child.wait()
            write(STATE/'child-exit.json',dict(exit_code=code,finished_unix=time.time()))
            if code!=0:raise RuntimeError(f'worker exit{code}; no automatic retry')
            status=json.loads((output/'status.json').read_text())
            if status.get('status')!='completed' or status.get('exit_code')!=0:raise ValueError('worker completion missing')
            check_registration(binding)
            save(status='awaiting_analysis',exit_code=0,model_output=str(output.relative_to(ROOT)))
        except Exception as error:
            save(status='failed',exit_code=1,error=repr(error));raise

if __name__=='__main__':main()
