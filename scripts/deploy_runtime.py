"""Install/start the bilingual text service from a verified local asset bundle."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from runtime_bundle import unpack, verify_installed

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/"runtime_state"
PIDS=STATE/"processes.json"


def record(path, payload):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    temp.replace(path)


def process_identity(pid):
    proc=Path("/proc")/str(pid)
    try:
        stat=(proc/"stat").read_text().rsplit(")",1)[1].split()
        if stat[0]=="Z":
            return None
        return dict(pid=pid,started=stat[19],cwd=os.readlink(proc/"cwd"),
                    command=[os.fsdecode(v) for v in (proc/"cmdline").read_bytes().split(b"\0") if v])
    except (FileNotFoundError, ProcessLookupError):
        return None


def stop(services):
    active=[]
    for item in services.values():
        current=process_identity(item["pid"])
        if current is None:
            continue
        if current!=item or current["cwd"]!=str(ROOT):
            raise RuntimeError(f"PID {item['pid']} does not match the recorded process")
        active.append(item)
    for item in active:
        # Recheck separately before signaling, in case a process exited during validation.
        if process_identity(item["pid"])==item:
            try:
                os.kill(item["pid"],signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline=time.monotonic()+20
    while any(process_identity(item["pid"])==item for item in active):
        if time.monotonic()>=deadline:
            raise RuntimeError("Recorded service still stopping; process file retained")
        time.sleep(.2)
    PIDS.unlink(missing_ok=True)


def port_free(port):
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try:
            listener.bind(("127.0.0.1",port))
        except OSError as exc:
            raise RuntimeError(f"Local port {port} is already in use or unavailable") from exc


def port_owned(pid,port):
    socket_ids=set()
    for fd in (Path("/proc")/str(pid)/"fd").iterdir():
        try:
            name=os.readlink(fd)
        except FileNotFoundError:
            continue
        if name.startswith("socket:["):
            socket_ids.add(name[8:-1])
    with Path("/proc/net/tcp").open() as handle:
        next(handle)
        return any(c[1].endswith(f":{port:04X}") and c[3]=="0A" and c[9] in socket_ids
                   for line in handle if (c:=line.split()))


def http(port,path,payload=None):
    data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode("utf-8")
    request=urllib.request.Request(f"http://127.0.0.1:{port}{path}",data=data,
                                   headers={"Content-Type":"application/json"} if data else {})
    # Local health and analysis traffic must stay off any environment proxy.
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request,timeout=150 if data else 3) as req:
        return req.read()


def ready(proc,port,path):
    end=time.monotonic()+60
    while time.monotonic()<end:
        if proc.poll() is not None:
            raise RuntimeError(f"Service on {port} exited with {proc.returncode}")
        try:
            if port_owned(proc.pid,port) and http(port,path):
                return
        except OSError:
            pass
        time.sleep(.3)
    raise TimeoutError(f"Service on {port} never became healthy")


def smoke(port,ui_port):
    if json.loads(http(port,"/health"))!={"status":"ok"} or http(ui_port,"/_stcore/health").strip()!=b"ok":
        raise RuntimeError("Health check failed")
    cases=json.loads((ROOT/"scripts/runtime_smoke_samples.json").read_text(encoding="utf-8"))
    evidence={}
    for group,text in cases.items():
        start=time.monotonic()
        body=json.loads(http(port,"/api/v1/analyze/text",{"text":text,"filename":f"portable-smoke-{group}.txt"}))
        detection=body["detection"]
        status=body["provenance"]["attribution"]
        if (body.get("modality")!="text" or detection["decision"]["threshold_group"]!=group
                or detection["label"]!="high_risk_ai" or not 0<=detection["score"]<=1
                or not body.get("content_id") or not body["report"].get("summary")
                or status["status"] not in ("ok","skipped")):
            raise RuntimeError(f"Invalid {group} smoke response: {detection.get('label')}, {status['status']}")
        if status["status"]=="ok":
            branch=next((b for b in status["branches"] if b.get("provider")=="window_a0_equal_fusion"),{})
            if (len(status["family_probabilities"])!=10 or len(status["top_k"])!=5
                    or set(branch.get("branch_weights",{}))!={"A0","B0","B1","B3"}
                    or status.get("recipe")!="window-A0+B0+B1+B3"):
                raise RuntimeError(f"Four-route attribution was not established for {group}")
        evidence[group]=dict(score=detection["score"],decision=detection["label"],id=body["content_id"],
            group=group,attribution=status["status"],recipe=status.get("recipe"),
            families=len(status.get("family_probabilities") or {}),seconds=round(time.monotonic()-start,3))
    if not any(r["attribution"]=="ok" for r in evidence.values()):
        raise RuntimeError("Both smoke inputs skipped text attribution")
    return evidence


def interpreter(venv, wheelhouse):
    python=venv/"bin/python"
    if python.is_file():
        return python
    if not wheelhouse or not wheelhouse.is_dir():
        raise RuntimeError("An existing --venv or a local --wheelhouse is required for offline Python dependencies")
    subprocess.run([sys.executable,"-m","venv",str(venv)],check=True)
    subprocess.run([str(python),"-m","pip","--disable-pip-version-check","install","--no-index",
                    "--find-links",str(wheelhouse),"-r",str(ROOT/"requirements-text-runtime.txt")],check=True)
    return python


def start(args):
    if PIDS.exists():
        raise RuntimeError("Existing runtime state. Run --stop before starting again")
    for port in (args.api_port,args.ui_port):
        port_free(port)
    python=interpreter(args.venv.resolve(),args.wheelhouse)
    verified=ROOT/"runtime_assets"
    if verified.exists():
        verify_installed(args.bundle,verified)
    else:
        unpack(args.bundle,verified)
    env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
             HF_HOME=str(STATE/"hf-cache"), HUGGINGFACE_HUB_CACHE=str(STATE/"hf-cache/hub"),
             HF_HUB_DISABLE_TELEMETRY="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
             OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS","2"),
             MKL_NUM_THREADS=os.environ.get("MKL_NUM_THREADS","2"),
             AIGC_CONFIG=str(STATE/"production-text.yaml"),
             AIGC_API_BASE=f"http://127.0.0.1:{args.api_port}/api/v1")
    env.pop("TRANSFORMERS_CACHE",None)
    env.pop("PYTHONPATH",None)
    if args.device=="cpu":
        env["CUDA_VISIBLE_DEVICES"]=""
    subprocess.run([str(python),"-m","scripts.configure_text_runtime",str(ROOT),args.device,
                    str(args.api_port),str(args.ui_port)], cwd=ROOT,env=env,check=True)
    commands={
        "api":[str(python),"-m","uvicorn","api.main:app","--host","127.0.0.1","--port",str(args.api_port)],
        "ui":[str(python),"-m","streamlit","run","api/dashboard.py",
            "--server.address","127.0.0.1","--server.port",str(args.ui_port),"--server.headless","true",
            "--browser.gatherUsageStats","false","--server.fileWatcherType","none"],
    }
    health={"api":(args.api_port,"/health"),"ui":(args.ui_port,"/_stcore/health")}
    services={}
    try:
        for name,argv in commands.items():
            with (STATE/f"{name}.log").open("ab") as log:
                proc=subprocess.Popen(argv,cwd=ROOT,env=env,start_new_session=True,
                                      stdin=subprocess.DEVNULL,stdout=log,stderr=log)
            info=process_identity(proc.pid)
            if info is None or info["command"]!=argv or info["cwd"]!=str(ROOT):
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=20)
                raise RuntimeError(f"Could not record {name} process identity")
            services[name]=info
            record(PIDS,services)
            ready(proc,*health[name])
        results=smoke(args.api_port,args.ui_port)
        record(STATE/"smoke.json",results)
        print(json.dumps(results,ensure_ascii=False,indent=2))
        print(f"API http://127.0.0.1:{args.api_port} · UI http://127.0.0.1:{args.ui_port}")
    except BaseException:
        stop(services)
        raise


def interrupted(signum, _frame):
    raise InterruptedError(f"Interrupted by signal {signum}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path)
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cpu")
    parser.add_argument("--api-port",type=int,default=8611)
    parser.add_argument("--ui-port",type=int,default=8612)
    parser.add_argument("--venv",type=Path,default=ROOT/".venv")
    parser.add_argument("--wheelhouse",type=Path)
    parser.add_argument("--stop",action="store_true")
    args=parser.parse_args()
    if not args.stop and (not args.bundle or not args.bundle.is_file()):
        parser.error("Startup requires a readable --bundle")
    if args.api_port==args.ui_port or any(not 1024<=p<=65535 for p in (args.api_port,args.ui_port)):
        parser.error("Provide two different unprivileged ports")
    os.umask(0o077)
    STATE.mkdir(exist_ok=True)
    for signum in (signal.SIGTERM,signal.SIGHUP):
        signal.signal(signum,interrupted)
    with (STATE/"lifecycle.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stop:
            if PIDS.exists():
                stop(json.loads(PIDS.read_text(encoding="utf-8")))
            print("Portable runtime stopped")
        else:
            start(args)


if __name__=="__main__":
    main()
