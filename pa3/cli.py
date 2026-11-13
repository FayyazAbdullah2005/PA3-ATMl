from __future__ import annotations
import argparse
import json
from pathlib import Path
from .io import Store,environment,write_json


COMMANDS=("doctor","prepare-models","prepare-benchmarks","task1-prepare","task1-train","task1-eval","task1-qualitative","task1-diagnostics","task1-evaluate-condition",
          "task2-prepare","task2-features","task2-train","task2-eval","task2-statistics","task3-prepare","task3-offline",
          "task3-adaptive","task3-live","quick-check","summarize","label-gallery","status","audit","verify-complete")
GPU_COMMANDS={"task1-train","task1-eval","task1-qualitative","task1-diagnostics","task2-features",
              "task2-train","task2-eval","task3-live","quick-check","task1-evaluate-condition"}


def execute(command,root="artifacts",run_id="main",method="full",condition="baseline",task="gsm8k_cot",
            tokens=196,smoke=False,source=None,commit=lambda:None):
    root=Path(root)
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+",run_id):
        raise ValueError("run-id must contain only letters, digits, underscores and hyphens")
    if condition not in ("baseline","full-mid","full","l2sp","lora4","lora16","oplora"):
        raise ValueError("Unknown Task 1 evaluation condition")
    if tokens not in (196,49):
        raise ValueError("Visual tokens must be 196 or 49")
    if command in GPU_COMMANDS and command!="quick-check":
        from .smoke import require_passed
        require_passed(root)
    if command=="doctor":
        import torch
        result={**environment(),"cuda_available":torch.cuda.is_available(),
                "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        print(json.dumps(result,indent=2))
        return result
    if command=="prepare-models":
        from .text import ensure_pins,cache_model
        from .config import QWEN,SIGLIP,LIVE_MODEL
        ensure_pins(root,commit)
        for name in (QWEN,SIGLIP,LIVE_MODEL):
            cache_model(root,name,commit)
        return
    if command=="task1-prepare":
        from .task1 import prepare
        return prepare(root,commit)
    if command=="prepare-benchmarks":
        from .benchmarks import prepare
        return prepare(root,commit)
    if command=="task1-train":
        from .task1 import train
        return train(root,method,run_id,smoke,commit)
    if command.startswith("task1-"):
        from .eval1 import benchmark,qualitative,diagnostics,evaluate_condition
        if command=="task1-eval":
            return benchmark(root,run_id,condition,task,commit,smoke)
        if command=="task1-qualitative":
            return qualitative(root,run_id,condition,commit)
        if command=="task1-evaluate-condition":
            return evaluate_condition(root,run_id,condition,commit)
        return diagnostics(root,run_id,condition,commit)
    if command=="task2-prepare":
        from .data2 import prepare
        return prepare(root,commit)
    if command=="task2-features":
        from .vision import cache_features
        return cache_features(root,commit)
    if command in ("task2-train","task2-eval","task2-statistics"):
        from .task2 import train,evaluate,statistics
        if command=="task2-statistics":
            return statistics(root,run_id,tokens,commit)
        return train(root,tokens,run_id,smoke,commit) if command=="task2-train" else evaluate(root,run_id,tokens,commit)
    if command.startswith("task3-"):
        from .task3 import prepare,offline,adaptive,live
        if command=="task3-prepare":
            return prepare(root,source,commit)
        if command=="task3-offline":
            return offline(root,run_id,commit)
        if command=="task3-adaptive":
            return adaptive(root,run_id,commit)
        return live(root,run_id,smoke,commit)
    if command=="quick-check":
        from .smoke import quick_check
        return quick_check(root,commit)
    if command=="summarize":
        from .reporting import summarize
        return summarize(root,run_id,commit)
    if command=="label-gallery":
        from .reporting import label_gallery
        label_gallery(root,root/"exports"/run_id)
        commit()
        return
    if command=="status":
        from collections import Counter
        units=Store(root).status()
        progress=[]
        for manifest in (root/"runs"/run_id).glob("task[12]/*/run.json"):
            from .io import read_json,read_events
            records=read_events(manifest.parent/"training.jsonl")
            last=records[-1] if records else {}
            progress.append({"run":str(manifest.parent.relative_to(root)),"logged_step":last.get("step",0),
                             "loss":last.get("sft_loss"),"training_seconds":last.get("training_seconds")})
        result={"completed_units":len(units),"by_area":dict(Counter(r["unit"].split("/")[0] for r in units)),
                "training":progress,"recent_units":sorted(units,key=lambda r:r["finished"] or 0)[-10:]}
        print(json.dumps(result,indent=2))
        return result
    if command=="audit":
        from .io import read_json,sha256
        failures=[]
        store=Store(root)
        for marker in root.rglob("DONE.json"):
            try:
                store.done(marker.parent)
            except Exception as error:
                failures.append({"path":str(marker.parent),"error":str(error)})
        for marker in root.rglob("step-*.json"):
            if marker.parent.name!="checkpoints":
                continue
            try:
                info=read_json(marker)
                checkpoint=(marker.parent/info["file"]).resolve()
                if not checkpoint.is_relative_to(marker.parent.resolve()):
                    raise RuntimeError("Checkpoint metadata escapes its directory")
                if not checkpoint.exists() or sha256(checkpoint)!=info["sha256"]:
                    raise RuntimeError("Permanent checkpoint checksum mismatch")
            except Exception as error:
                failures.append({"path":str(marker),"error":str(error)})
        write_json(root/"audit.json",{"failures":failures,"units":len(store.status())})
        commit()
        if failures:
            raise RuntimeError(f"Integrity audit failed for {len(failures)} units; see audit.json")
        print("All completed artifact checksums verified.")
        return
    if command=="verify-complete":
        from .config import T1_METHODS,T1_TASKS
        from .task2 import development_steps
        from .io import read_json
        source=root/"runs"/run_id
        conditions=("baseline","full-mid",*T1_METHODS)
        paths=[source/"task1"/method for method in T1_METHODS]
        paths += [source/"task1-evaluation"/condition/task for condition in conditions for task in T1_TASKS]
        paths += [source/"task1-qualitative"/condition for condition in conditions]
        paths += [source/"task1-diagnostics"/condition for condition in conditions if condition!="baseline"]
        for count in (196,49):
            paths += [source/"task2"/str(count),source/"task2-evaluation"/str(count),source/"task2-evaluation"/str(count)/"statistics"]
            paths += [source/"task2"/str(count)/"development"/f"step-{step:06d}" for step in development_steps()]
        paths += [source/name for name in ("task3-offline","task3-adaptive","task3-live")]
        store=Store(root)
        missing=[str(path.relative_to(root)) for path in paths if not store.done(path)]
        inventory=root/"exports"/run_id/"inventory.json"
        if not inventory.exists() or not read_json(inventory)["complete_counts"]:
            missing.append("Exported evidence counts are incomplete")
        write_json(source/"completion.json",{"complete":not missing,"missing":missing})
        commit()
        if missing:
            raise RuntimeError(f"{len(missing)} experiment/evidence units are incomplete; see completion.json")
        print("All required computational evidence, including the optional 49-token run, is complete.")
        return
    raise ValueError(command)


def main():
    parser=argparse.ArgumentParser(description="ATML PA3 reproducible experiment runner")
    parser.add_argument("command",choices=COMMANDS)
    parser.add_argument("--root",default="artifacts")
    parser.add_argument("--run-id",default="main")
    parser.add_argument("--method",choices=("full","l2sp","lora4","lora16","oplora"),default="full")
    parser.add_argument("--condition",default="baseline")
    parser.add_argument("--task",choices=("gsm8k_cot","hellaswag","boolq","arc_easy"),default="gsm8k_cot")
    parser.add_argument("--tokens",type=int,choices=(196,49),default=196)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--source",help="Already downloaded ATML-PA3-Data folder")
    args=vars(parser.parse_args())
    import os,sys
    if os.environ.get("PA3_COMMIT_PROTOCOL")=="1":
        token=os.environ["PA3_COMMIT_TOKEN"]
        def commit():
            print("\n__PA3_COMMIT__:"+token,flush=True)
            if sys.stdin.readline().strip()!="COMMIT_OK:"+token:
                raise RuntimeError("Persistence acknowledgement missing; stopping before more expensive work")
        execute(**args,commit=commit)
    else:
        execute(**args)

