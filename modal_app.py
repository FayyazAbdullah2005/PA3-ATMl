"""Windows launcher: all heavy libraries and all data remain inside Modal."""
from pathlib import Path
import os
import modal

APP_NAME="atml-pa3"
VOLUME_NAME="atml-pa3-artifacts"
ROOT="/vol"
app=modal.App(APP_NAME)
volume=modal.Volume.from_name(VOLUME_NAME,create_if_missing=True,version=2)
image=(modal.Image.debian_slim(python_version="3.11")
       .apt_install("libgl1","libglib2.0-0")
       .pip_install_from_requirements("requirements.txt")
       .env({"HF_HOME":"/vol/data/hf","HF_DATASETS_CACHE":"/vol/data/hf/datasets",
             "TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"4",
             "MPLCONFIGDIR":"/tmp/matplotlib","PYTHONUNBUFFERED":"1",
             "PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"})
       .add_local_python_source("pa3")
       .add_local_dir("tests",remote_path="/root/tests"))


def child_job(command,options,offline=False):
    """Run numerical work in a real main thread (Math-Verify uses signal alarms).

    A commit/ack pipe pauses the child until the parent has durably committed the
    Volume, eliminating a race with the child's next open output file.
    """
    # Warm workers retain an older mount snapshot. Refresh before the child opens
    # any files so CPU statistics can see the preceding GPU job's committed output.
    volume.reload()
    import sys
    from pa3.process import run_with_commits
    if command=="tests":
        args=[sys.executable,"-m","unittest","discover","-s","/root/tests","-t","/root","-v"]
    else:
        args=[sys.executable,"-m","pa3",command,"--root",ROOT]
        for key,value in options.items():
            if isinstance(value,bool):
                if value:
                    args.append("--"+key.replace("_","-"))
            elif value is not None:
                args.extend(["--"+key.replace("_","-"),str(value)])
    env={}
    if offline:
        env.update(HF_HUB_OFFLINE="1",HF_DATASETS_OFFLINE="1")
    status=run_with_commits(args,"/root",env,volume.commit)
    volume.commit()
    if status:
        raise RuntimeError(f"{command} failed with exit code {status}. Completed units and verified recovery checkpoints "
            "remain on the Volume. Inspect the preceding traceback; for CUDA out-of-memory, rerun with --gpu A100-40GB.")


@app.function(image=image,volumes={ROOT:volume},cpu=4,memory=16384,
              timeout=86400,max_containers=1,retries=0)
def cpu_job(command:str,options:dict):
    child_job(command,options)
    volume.commit()
    return {"completed":command,"options":options}


@app.function(image=image,volumes={ROOT:volume},gpu="A10G",cpu=4,memory=32768,
              timeout=86400,max_containers=1,retries=0,scaledown_window=2)
def gpu_job(command:str,options:dict):
    child_job(command,options,offline=True)
    return {"completed":command,"options":options}


@app.function(image=image,volumes={ROOT:volume},cpu=.125,memory=512,
              timeout=86400,max_containers=1,retries=0)
def orchestrate(command:str,options:dict,gpu:str):
    volume.reload()
    from pa3.cli import GPU_COMMANDS
    from pa3.config import T1_METHODS
    from pa3.io import Store,read_json,write_json,digest,code_hash,sha256,environment
    import json,time
    events=Path(ROOT)/"orchestration.jsonl"
    def event(value):
        # Keep orchestration state visible across detached calls and failures.
        with events.open("a",encoding="utf-8") as f:
            f.write(json.dumps({"time":time.time(),**value})+"\n")
        volume.commit()
        print(value,flush=True)
    def run(name,**overrides):
        settings={**options,**overrides}
        def output_markers():
            root=Path(ROOT)
            run=root/"runs"/settings["run_id"]
            smoke=root/"runs"/"smoke"
            condition=settings["condition"]
            if name=="quick-check": return [root/"checks"/"gpu"]
            if name=="task1-train": return [(smoke if settings["smoke"] else run)/"task1"/settings["method"]]
            if name=="task1-evaluate-condition":
                return [run/"task1-evaluation"/condition/t for t in ("gsm8k_cot","hellaswag","boolq","arc_easy")]+[run/"task1-qualitative"/condition]
            if name=="task1-eval": return [(root/"runs"/"eval-smoke" if settings["smoke"] else run)/"task1-evaluation"/condition/settings["task"]]
            if name=="task1-qualitative": return [run/"task1-qualitative"/condition]
            if name=="task1-diagnostics": return [root/"data"/"activation-probe" if condition=="baseline" else run/"task1-diagnostics"/condition]
            if name=="task2-features": return [root/"data"/"task2"/"features"]
            if name=="task2-train": return [(smoke if settings["smoke"] else run)/"task2"/str(settings["tokens"])]
            if name=="task2-eval": return [run/"task2-evaluation"/str(settings["tokens"])]
            if name=="task3-live": return [(root/"runs"/"live-smoke" if settings["smoke"] else run)/"task3-live"]
            return []
        receipt=None
        receipt_signature=None
        if name in GPU_COMMANDS:
            pins=Path(ROOT)/"data"/"pins.json"
            code=code_hash("config.py","io.py","train.py","models.py","text.py","task1.py","eval1.py",
                           "vision.py","task2.py","task3.py","scoring.py","smoke.py","benchmarks.py")
            dependencies=[]
            if name.startswith("task1-"):
                dependencies=[Path(ROOT)/"data"/"task1"/"DONE.json",Path(ROOT)/"data"/"benchmarks"/"DONE.json"]
                if name!="task1-train" and settings["condition"]!="baseline":
                    method="full" if settings["condition"]=="full-mid" else settings["condition"]
                    step=625 if settings["condition"]=="full-mid" else 1250
                    dependencies.append(Path(ROOT)/"runs"/settings["run_id"]/"task1"/method/"checkpoints"/f"step-{step:06d}.json")
            elif name.startswith("task2-"):
                dependencies=[Path(ROOT)/"data"/"task2"/"DONE.json"]
                if name!="task2-features":
                    dependencies.append(Path(ROOT)/"data"/"task2"/"features"/"DONE.json")
                if name=="task2-eval":
                    dependencies.append(Path(ROOT)/"runs"/settings["run_id"]/"task2"/str(settings["tokens"])/"DONE.json")
            elif name=="task3-live":
                dependencies=[Path(ROOT)/"data"/"task3"/"DONE.json"]
            dependency_hashes={str(p):sha256(p) if p.exists() else None for p in dependencies}
            receipt_signature=digest([name,settings,code,environment()["packages"],
                                      sha256(pins) if pins.exists() else None,dependency_hashes])
            receipt=Path(ROOT)/"jobs"/receipt_signature
            store=Store(ROOT,volume.commit)
            if store.done(receipt,receipt_signature):
                saved=read_json(receipt/"outputs.json")
                if all(Path(p).exists() and sha256(Path(p))==expected and store.done(Path(p).parent) for p,expected in saved.items()):
                    event({"reused_without_gpu":name,"options":settings})
                    if name=="task2-eval":
                        run("task2-statistics",tokens=settings["tokens"])
                    return
            targets=output_markers()
            if targets and all(store.done(p) for p in targets):
                write_json(receipt/"outputs.json",{str(p/"DONE.json"):sha256(p/"DONE.json") for p in targets})
                store.finish(receipt,receipt_signature,[receipt/"outputs.json"])
                event({"reused_without_gpu":name,"options":settings})
                if name=="task2-eval":
                    run("task2-statistics",tokens=settings["tokens"])
                return
        event({"starting":name,"options":settings})
        if name in GPU_COMMANDS:
            result=gpu_job.with_options(gpu=gpu).remote(name,settings)
        else:
            result=cpu_job.remote(name,settings)
        volume.reload()
        if receipt is not None:
            targets=output_markers()
            if not targets or not all(Store(ROOT).done(p) for p in targets):
                raise RuntimeError(f"{name} returned without verified completed output units")
            write_json(receipt/"outputs.json",{str(p/"DONE.json"):sha256(p/"DONE.json") for p in targets})
            Store(ROOT,volume.commit).finish(receipt,receipt_signature,[receipt/"outputs.json"])
        event(result)
        if name=="task2-eval":
            run("task2-statistics",tokens=settings["tokens"])
    def prepare():
        run("prepare-models")
        run("prepare-benchmarks")
        for name in ("task1-prepare","task2-prepare","task3-prepare"):
            run(name)
        run("label-gallery")
    if command=="prepare":
        prepare()
        return
    if command=="quick-check":
        run("prepare-models")
        run("tests")
        run("quick-check")
        return
    if command=="all":
        prepare()
        run("tests")
        run("task3-offline")
        run("task3-adaptive")
        run("quick-check")
        run("task1-evaluate-condition",condition="baseline")
        run("task1-diagnostics",condition="baseline")
        for method in T1_METHODS:
            run("task1-train",method=method)
        for condition in ("full-mid",*T1_METHODS):
            run("task1-evaluate-condition",condition=condition)
            run("task1-diagnostics",condition=condition)
        run("task2-features")
        run("task2-train",tokens=196)
        run("task2-eval",tokens=196)
        run("task3-live")
        run("task2-train",tokens=49)
        run("task2-eval",tokens=49)
        run("summarize")
        run("audit")
        run("verify-complete")
        event({"all_complete":True,"run_id":options["run_id"]})
        return
    run(command)


@app.local_entrypoint()
def main(command:str="status",run_id:str="main",method:str="full",condition:str="baseline",
         task:str="gsm8k_cot",tokens:int=196,smoke:bool=False,gpu:str="A10G"):
    """Launch with modal run --detach modal_app.py --command ... ."""
    from pa3.cli import COMMANDS
    if command not in (*COMMANDS,"prepare","all","tests"):
        raise ValueError(f"Unknown command: {command}")
    if gpu not in ("A10G","A100-40GB","A100-80GB","L40S","H100"):
        raise ValueError("Unsupported GPU choice")
    if tokens not in (196,49):
        raise ValueError("tokens must be 196 or 49")
    if command=="all" and smoke:
        raise ValueError("Use quick-check for reduced tests; --smoke cannot produce a complete assignment pipeline")
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("run-id must contain only letters, digits, underscores, and hyphens")
    options={"run_id":run_id,"method":method,"condition":condition,"task":task,"tokens":tokens,"smoke":smoke}
    # remote() is intentional: --detach keeps this app alive when the laptop closes.
    # Remote failures must reach the CLI as failures; --detach itself handles
    # laptop disconnects and does not make a failed orchestrator keep running.
    result=orchestrate.remote(command,options,gpu)
    print(result if result is not None else "Command completed. See Modal logs and persistent Volume.")

