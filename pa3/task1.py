"""Task 1 preparation and five controlled training methods."""
from __future__ import annotations
import json
from pathlib import Path
import time
import unicodedata
import numpy as np
import torch
from .config import *
from .io import *
from .models import *
from .text import *
from .train import *


def normalize_question(value):
    return " ".join(unicodedata.normalize("NFKC", value or "").casefold().split())


def prepare(root, commit=lambda: None):
    from datasets import load_dataset
    import yaml
    root = Path(root)
    store = Store(root, commit)
    pins = ensure_pins(root, commit)
    d = root / "data" / "task1"
    sig = digest({"pins": pins, "examples": 40000,
                  "code": code_hash("task1.py", "text.py", "config.py")})
    if store.done(d, sig):
        print("Task 1 prepared data verified; reusing it.")
        return
    d.mkdir(parents=True, exist_ok=True)
    cache_model(root, QWEN, commit)
    tokenizer = tokenizer_for(root)
    gsm = load_dataset("openai/gsm8k", "main", split="test", revision=pins["datasets"]["openai/gsm8k"])
    forbidden = {normalize_question(x["question"]) for x in gsm}
    source = load_dataset("meta-math/MetaMathQA", split="train", streaming=True,
                          revision=pins["datasets"]["meta-math/MetaMathQA"])
    rejected = {"empty": 0, "gsm8k_overlap": 0, "length": 0}
    # The selection itself can be resumed without re-tokenizing retained examples.
    staging = d / "selection.jsonl"
    retained = read_events(staging)
    if retained:
        old = read_json(d / "selection_signature.json")
        if old["signature"] != sig:
            raise RuntimeError("Partial data selection belongs to a different implementation")
        # Recover rejection counts and source position from the last retained row.
        rejected = dict(retained[-1]["rejected_so_far"])
    else:
        write_json(d / "selection_signature.json", {"signature": sig})
    after = retained[-1]["source_id"] if retained else -1
    if len(retained) < 40000:
        for source_id, row in enumerate(source):
            if source_id <= after:
                continue
            question, answer = row.get("query", ""), row.get("response", "")
            if not str(question).strip() or not str(answer).strip():
                rejected["empty"] += 1
                continue
            if any(normalize_question(row.get(key)) in forbidden for key in ("query", "original_question")):
                rejected["gsm8k_overlap"] += 1
                continue
            example = supervised_example(tokenizer, question, answer)
            if len(example["input_ids"]) > 2048:
                rejected["length"] += 1
                continue
            example.update(source_id=source_id, question=question, response=answer,
                           rejected_so_far=dict(rejected))
            append_event(staging, example)
            retained.append(example)
            if len(retained) % 1000 == 0:
                print(f"Selected {len(retained)}/40000 examples", flush=True)
                commit()
            if len(retained) == 40000:
                break
    if len(retained) != 40000:
        raise RuntimeError(f"Only {len(retained)} eligible examples; refusing a reduced training budget")
    write_parquet(d / "train.parquet", retained)
    hellaswag = load_dataset("Rowan/hellaswag", split="validation", revision=pins["datasets"]["Rowan/hellaswag"])
    rng = np.random.default_rng(SEED)
    probe = {"gsm8k": rng.choice(len(gsm), 32, replace=False).tolist(),
             "hellaswag": rng.choice(len(hellaswag), 32, replace=False).tolist()}
    write_json(d / "probe_ids.json", probe)
    # HellaSwag probe formatting is reconstructed from the pinned harness at evaluation.
    write_json(d / "probe_records.json", {"gsm8k": [dict(gsm[i]) for i in probe["gsm8k"]],
                                         "hellaswag": [dict(hellaswag[i]) for i in probe["hellaswag"]]})
    download(QUALITATIVE_URL, d / "first_plot_questions.yaml")
    qualitative = yaml.safe_load((d / "first_plot_questions.yaml").read_text(encoding="utf-8"))
    if len(qualitative) != 24:
        raise ValueError("Expected exactly 24 qualitative prompts")
    write_json(d / "selection.json", {"source_ids": [r["source_id"] for r in retained],
               "rejected": rejected, "seed": SEED, "normalization": "NFKC + casefold + whitespace"})
    store.finish(d, sig, [d / f for f in ("train.parquet", "selection.json", "probe_ids.json",
        "probe_records.json", "first_plot_questions.yaml")])


def run_configuration(root, method, smoke=False):
    d = Path(root) / "data" / "task1"
    if not Store(root).done(d):
        raise RuntimeError("Run task1-prepare first")
    return {"task": 1, "method": method, **T1_METHODS[method], "seed": SEED,
            "shared":T1_SHARED,"pins":read_json(Path(root)/"data"/"pins.json"),
            "smoke": smoke, "updates": 2 if smoke else 1250,
            "data": read_json(d / "DONE.json")["signature"],
            "code": code_hash("task1.py", "models.py", "train.py", "text.py", "config.py")}


def train(root, method, run_id="main", smoke=False, commit=lambda: None):
    if method not in T1_METHODS:
        raise ValueError(f"Unknown method {method}")
    root = Path(root)
    configuration = run_configuration(root, method, smoke)
    directory = root / "runs" / ("smoke" if smoke else run_id) / "task1" / method
    store = Store(root, commit)
    sig = digest(configuration)
    if store.done(directory, sig):
        try:
            Checkpoints(directory/"checkpoints",sig).latest(configuration["updates"])
            print(f"{method}: verified completed run; no GPU work needed.")
            return
        except FileNotFoundError:
            print("Final checkpoint is unavailable; resuming the latest verified recovery state.",flush=True)
    seed_all(SEED)
    cfg = T1_METHODS[method]
    # Derive OPLoRA's bases from canonical FP32 starting weights before the
    # backbone's prescribed FP16 cast; do not SVD a previously rounded weight.
    model=load_qwen(root,full=(not cfg["rank"] or method=="oplora"),training=True)
    references = None
    reference_begin = time.perf_counter()
    if cfg["l2"]:
        references = {n: p.detach().float().clone() for n, p in unique_parameters(model)}
    reference_seconds=time.perf_counter()-reference_begin
    basis_meta = []
    if cfg["rank"]:
        model.requires_grad_(False)
        bases = None
        if method == "oplora":
            bases, basis_meta = compute_bases(model, TARGETS,
                root / "data" / "bases" / read_json(root/"data"/"pins.json")["models"][QWEN],
                commit=commit)
            model.half()
        install_lora(model, TARGETS, cfg["rank"], cfg["alpha"], bases=bases)
        model.enable_input_require_grads()
    for name, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            raise TypeError(f"Trainable parameter is not FP32: {name}")
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg["lr"]), betas=(.9, .999), eps=1e-8)
    trainer = Trainer(model, optimizer, directory, configuration, commit, full_state=not cfg["rank"])
    trainer.restore()
    rows = read_parquet(root / "data" / "task1" / "train.parquet")
    tokenizer = tokenizer_for(root)
    # Smoke uses the same microbatch/accumulation; select short records to minimize GPU time.
    if smoke:
        rows = sorted(rows[:512], key=lambda x: len(x["input_ids"]))[:64]
    def batches(step):
        examples = rows[step*32:(step+1)*32]
        if len(examples) != 32:
            raise ValueError("Effective batch must contain exactly 32 examples")
        return [collate_text(examples[i:i+2], tokenizer.pad_token_id) for i in range(0, 32, 2)]
    def forward(batch):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        labels = batch.pop("labels")
        output = model(**batch, use_cache=False)
        return token_losses(output.logits, labels)
    total = configuration["updates"]
    resources = {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "reference_bytes": sum(p.numel()*p.element_size() for p in (references or {}).values()),
        "reference_setup_seconds": reference_seconds,
        "basis_bytes": sum(x["bytes"] for x in basis_meta),
        "basis_preprocessing_seconds": sum(x["seconds"] for x in basis_meta)}
    write_json(directory / "resources.json", resources)
    trainer.run(total, batches, forward,
        lambda step: cfg["lr"]*schedule_factor(step+1, 1250, 38),
        references=references, l2=cfg["l2"], milestones=() if smoke else T1_MILESTONES,
        save_every=25 if cfg["rank"] else 125)
    files=[directory/"run.json",directory/"resources.json",directory/"completed.json"]
    if cfg["rank"]:
        export_begin=time.perf_counter()
        factors={}
        with torch.no_grad():
            for name,module in model.named_modules():
                if isinstance(module,LoRALinear):
                    a,b=module.factors()
                    factors[name]={"A":a.cpu(),"B":b.cpu(),"alpha":cfg["alpha"],"rank":cfg["rank"]}
        with atomic_path(directory/"adapter_factors.pt") as tmp:
            torch.save(factors,tmp)
        resources["export_seconds"]=time.perf_counter()-export_begin
        resources["export_bytes"]=(directory/"adapter_factors.pt").stat().st_size
        resources["export_projected"]=method=="oplora"
        write_json(directory/"resources.json",resources)
        files.append(directory/"adapter_factors.pt")
    write_json(directory / "completed.json", {"updates": trainer.step, "training_seconds": trainer.elapsed})
    resources.update(optimizer_seconds=trainer.elapsed,data_seconds=trainer.extra.get("data_seconds",0),
        checkpoint_seconds=sum(r["seconds"] for r in read_events(directory/"checkpoint_times.jsonl")),
        amp_skipped_attempts=len(read_events(directory/"overflow.jsonl")))
    resources["hardware_segments"]=trainer.extra.get("hardware_segments",[])
    resources["training_wall_seconds"]=resources["optimizer_seconds"]+resources["data_seconds"]+resources["checkpoint_seconds"]
    write_json(directory/"resources.json",resources)
    store.finish(directory,sig,files)

