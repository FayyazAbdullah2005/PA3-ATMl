"""Pinned stock harness tasks, prepared on CPU before GPU scoring."""
from pathlib import Path
from .config import T1_TASKS
from .io import Store,digest,read_json,write_json,code_hash

REPOSITORIES={"gsm8k_cot":"openai/gsm8k","hellaswag":"Rowan/hellaswag",
              "boolq":"aps/super_glue","arc_easy":"allenai/ai2_arc"}
TASK_FILES={"gsm8k_cot":"gsm8k/gsm8k-cot.yaml","hellaswag":"hellaswag/hellaswag.yaml",
            "boolq":"super_glue/boolq/default.yaml","arc_easy":"arc/arc_easy.yaml"}


def manager_for(names):
    import lm_eval
    from lm_eval.tasks import TaskManager
    base=Path(lm_eval.__file__).parent/"tasks"
    directories=sorted({str((base/TASK_FILES[name]).parent) for name in names})
    # Do not scan thousands of unrelated harness tasks inside a paid GPU job.
    return TaskManager(include_defaults=False,include_path=directories)


def tasks(root,names):
    from lm_eval.tasks import TaskManager,get_task_dict
    manager=manager_for(names)
    pins=read_json(Path(root)/"data"/"pins.json")
    configurations=[{"task":name,"dataset_path":REPOSITORIES[name],
        "dataset_kwargs":{"revision":pins["datasets"][REPOSITORIES[name]]}} for name in names]
    return get_task_dict(configurations,task_manager=manager)


def prepare(root,commit=lambda:None):
    from .text import ensure_pins
    root=Path(root)
    pins=ensure_pins(root,commit)
    d=root/"data"/"benchmarks"
    sig=digest({"pins":pins,"code":code_hash("benchmarks.py")})
    store=Store(root,commit)
    if store.done(d,sig):
        return
    objects=tasks(root,list(T1_TASKS))
    records={}
    for name,task in objects.items():
        docs=task.eval_docs
        if len(docs)!=T1_TASKS[name][1]:
            raise ValueError(f"Unexpected {name} split size: {len(docs)}")
        records[name]={"configuration":task.config.to_dict(),"examples":len(docs),
                       "fingerprint":getattr(docs,"_fingerprint",None)}
    write_json(d/"tasks.json",records)
    store.finish(d,sig,[d/"tasks.json"])

