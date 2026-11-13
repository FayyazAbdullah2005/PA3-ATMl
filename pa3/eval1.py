"""Pinned harness scoring with persistent request caches, and Task 1 diagnostics."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import torch
from .config import *
from .io import *
from .models import *
from .text import *
from .train import Checkpoints, rng_state, restore_rng


def condition_checkpoint(root, run_id, condition):
    if condition == "baseline":
        return None
    method, step = ("full", 625) if condition == "full-mid" else (condition, 1250)
    directory = Path(root)/"runs"/run_id/"task1"/method
    run = read_json(directory/"run.json")
    checkpoint = Checkpoints(directory/"checkpoints", run["signature"]).latest(step)
    return checkpoint[0],method,checkpoint[1].get("model_sha256",checkpoint[1]["sha256"])


def load_condition(root, run_id, condition, commit=lambda: None, merge=True):
    info = condition_checkpoint(root, run_id, condition)
    model = load_qwen(root, full=True)
    if info is None:
        return model.half().eval()
    path, method, _ = info
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = T1_METHODS[method]
    if not cfg["rank"]:
        model.load_state_dict(state["model"], strict=True)
    else:
        model.requires_grad_(False)
        install_lora(model,TARGETS,cfg["rank"],cfg["alpha"])
        # Inference consumes exported factors; OPLoRA's export is already projected.
        run_directory=Path(root)/"runs"/run_id/"task1"/method
        if not Store(root).done(run_directory):
            raise RuntimeError("Adapter run has no verified completed export")
        factors=torch.load(run_directory/"adapter_factors.pt",map_location="cpu",weights_only=True)
        state["model"]={name+"."+factor:values[factor] for name,values in factors.items() for factor in ("A","B")}
        # Checkpoint state is named relative to the original Qwen module.
        for n, p in model.named_parameters():
            if n in state["model"]:
                p.requires_grad_(True)
        load_trainable(model, state["model"])
        model.eval()
        if merge:
            tok = tokenizer_for(root)
            ids = torch.tensor([chat_ids(tok, [{"role": "user", "content": "What is 2 + 2?"}], True)], device="cuda")
            with torch.inference_mode():
                before = model(ids).logits[:, -1].float()
                with torch.autocast("cuda",dtype=torch.float16):
                    before_fp16=model(ids).logits[:,-1].float()
                merge_lora(model)
                after = model(ids).logits[:, -1].float()
            error = float((before-after).abs().max())
            if not torch.allclose(before, after, atol=2e-4, rtol=2e-4):
                raise RuntimeError(f"Adapter merge failed FP32 logit check (max error={error})")
            model.half()
            with torch.inference_mode():
                after_fp16=model(ids).logits[:,-1].float()
            fp16_error=float((before_fp16-after_fp16).abs().max())
            if not torch.isfinite(after_fp16).all() or fp16_error>.25:
                raise RuntimeError(f"FP16 merged/unmerged logits differ excessively: {fp16_error}")
            out = Path(root)/"runs"/run_id/"task1"/method/"merge_check.json"
            write_json(out,{"max_abs_logit_error_fp32":error,"max_abs_logit_error_fp16":fp16_error,
                           "fp16_argmax_agrees":bool(before_fp16.argmax()==after_fp16.argmax())})
    return model.half().eval()


def persistent_hflm(model, tokenizer, directory, commit, batch_size=16):
    from lm_eval.models.huggingface import HFLM
    directory = Path(directory)
    class PersistentHFLM(HFLM):
        def __init__(self):
            super().__init__(pretrained=model, tokenizer=tokenizer, batch_size=batch_size,
                             max_length=4096, dtype="float16", enable_thinking=False,
                             chat_template_args={"enable_thinking": False})
            self.cached = {r["key"]: r["value"] for r in read_events(directory/"requests.jsonl")}
            self.decode_info = {r["key"]: r for r in read_events(directory/"decode.jsonl")}

        def _model_generate(self, context, max_length, stop, **kwargs):
            output = super()._model_generate(context, max_length, stop, **kwargs)
            pad = self.tokenizer.pad_token_id
            records = []
            for i in range(context.shape[0]):
                prompt = context[i].tolist()
                while prompt and prompt[0] == pad:
                    prompt.pop(0)
                tokens = output[i, context.shape[1]:].tolist()
                # Generation pads shorter finished rows to the longest row.
                eos = self.model.generation_config.eos_token_id
                stops = set(eos if isinstance(eos, list) else [eos])
                end = next((j+1 for j, t in enumerate(tokens) if t in stops), len(tokens))
                tokens = tokens[:end]
                record = {"key": digest(prompt), "input_tokens": len(prompt),
                          "output_tokens": len(tokens), "generated_token_ids": tokens,
                          "hit_max_new_tokens":len(tokens)>=max_length-context.shape[1]
                            and (not tokens or tokens[-1] not in stops)
                            and not any(s and s in self.tokenizer.decode(tokens,skip_special_tokens=False) for s in stop)}
                self.decode_info[record["key"]] = record
                records.append(record)
            append_events(directory/"decode.jsonl", records)
            return output

        def cached_call(self, kind, requests, operation):
            keys = [digest({"kind": kind, "args": request.args}) for request in requests]
            missing = [(key, request) for key, request in zip(keys, requests) if key not in self.cached]
            chunk_size = 32 if kind == "gen" else 512
            for start in range(0, len(missing), chunk_size):
                chunk = missing[start:start+chunk_size]
                values = operation([request for _, request in chunk])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if len(values) != len(chunk):
                    raise RuntimeError("Harness returned the wrong number of request results")
                events = []
                for (key, request), value in zip(chunk, values):
                    self.cached[key] = value
                    events.append({"key": key, "value": value})
                append_events(directory/"requests.jsonl", events)
                # Persist every chunk, including raw token/termination diagnostics.
                commit()
            return [tuple(self.cached[key]) if kind=="ll" else self.cached[key] for key in keys]

        def loglikelihood(self, requests, disable_tqdm=False):
            return self.cached_call("ll", requests,
                lambda chunk: super(PersistentHFLM, self).loglikelihood(chunk, disable_tqdm=True))

        def generate_until(self, requests, disable_tqdm=False):
            return self.cached_call("gen", requests,
                lambda chunk: super(PersistentHFLM, self).generate_until(chunk, disable_tqdm=True))
    return PersistentHFLM()


def benchmark(root, run_id, condition, task_name, commit=lambda: None, smoke=False, loader=None):
    import lm_eval
    root = Path(root)
    if task_name not in T1_TASKS:
        raise ValueError(task_name)
    checkpoint = condition_checkpoint(root, run_id, condition)
    signature = digest({"condition": condition, "checkpoint": checkpoint[2] if checkpoint else "original",
        "pins": read_json(root/"data"/"pins.json"), "task": task_name, "smoke": smoke,
        "code": code_hash("eval1.py", "text.py", "models.py"), "lm_eval": environment()["packages"]["lm-eval"]})
    d = root/"runs"/("eval-smoke" if smoke else run_id)/"task1-evaluation"/condition/task_name
    store = Store(root, commit)
    if store.done(d, signature):
        print(f"Reusing completed {condition}/{task_name}")
        return
    d.mkdir(parents=True, exist_ok=True)
    model,tokenizer = loader() if loader else (load_condition(root,run_id,condition,commit),tokenizer_for(root))
    lm = persistent_hflm(model, tokenizer, d, commit, batch_size=16 if task_name == "gsm8k_cot" else (2 if task_name == "boolq" else 8))
    from .benchmarks import tasks,manager_for
    task_object=tasks(root,[task_name])[task_name]
    result = lm_eval.simple_evaluate(model=lm,tasks=[task_object],num_fewshot=0,
        apply_chat_template=task_name == "gsm8k_cot", log_samples=True,
        limit=2 if smoke else None, bootstrap_iters=0,
        gen_kwargs={"max_gen_toks": 1024, "do_sample": False} if task_name == "gsm8k_cot" else None,
        random_seed=SEED,numpy_random_seed=SEED,torch_random_seed=SEED,fewshot_random_seed=SEED,
        task_manager=manager_for([task_name]))
    samples = result.pop("samples")[task_name]
    raw_samples=samples
    if task_name=="gsm8k_cot":
        samples=[s for s in samples if s["filter"]=="flexible-extract"]
    if not smoke and len(samples) != T1_TASKS[task_name][1]:
        raise ValueError(f"Incomplete {task_name} split: {len(samples)}")
    if task_name == "gsm8k_cot":
        for sample in samples:
            # Harness stores each Instance's positional arguments in order.
            prompt = sample["arguments"][0][0]
            key = digest(lm.tok_encode(prompt))
            info = lm.decode_info.get(key)
            if info is None:
                raise RuntimeError("Could not associate GSM8K token diagnostics with its prompt; raw generation cache remains reusable")
            sample.update({k: v for k, v in info.items() if k != "key"})
            sample["extracted_answer"]=sample["filtered_resps"][0]
    del lm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # JSON keeps heterogeneous harness fields readable and lossless.
    write_json(d/"scores.json", result)
    write_json(d/"examples.json", samples)
    write_json(d/"all_filters.json",raw_samples)
    store.finish(d,signature,[d/"scores.json",d/"examples.json",d/"all_filters.json"])


def qualitative(root, run_id, condition, commit=lambda: None, loader=None):
    import yaml
    root = Path(root)
    checkpoint = condition_checkpoint(root, run_id, condition)
    sig = digest({"condition": condition, "checkpoint": checkpoint[2] if checkpoint else "original",
        "prompts": sha256(root/"data"/"task1"/"first_plot_questions.yaml"),
        "code": code_hash("eval1.py", "text.py"), "max_new_tokens": 512})
    d = root/"runs"/run_id/"task1-qualitative"/condition
    store = Store(root, commit)
    unit_store = Store(root)
    if store.done(d, sig):
        return
    prompts = yaml.safe_load((root/"data"/"task1"/"first_plot_questions.yaml").read_text(encoding="utf-8"))
    model, tok = None, tokenizer_for(root)
    records = []
    generated_new = 0
    for p in prompts:
        unit = d/p["id"]
        usig = digest([sig, p["id"]])
        if not unit_store.done(unit, usig):
            if model is None:
                if loader:
                    model,tok=loader()
                else:
                    model=load_condition(root,run_id,condition,commit)
            messages = ([{"role": "system", "content": p["system"]}] if p.get("system") else [])
            messages.append({"role": "user", "content": p["paraphrases"][0]})
            row = generate_text(model, tok, messages, 512)
            row.update(prompt_id=p["id"], condition=condition)
            write_json(unit/"response.json", row)
            unit_store.finish(unit, usig, [unit/"response.json"])
            generated_new += 1
            if generated_new % 8 == 0:
                commit()
        records.append(read_json(unit/"response.json"))
    write_json(d/"responses.json", records)
    store.finish(d, sig, [d/"responses.json"])


def evaluate_condition(root,run_id,condition,commit=lambda:None):
    """Load one checkpoint once for all benchmarks and qualitative generations."""
    bundle=[]
    def loader():
        if not bundle:
            bundle.extend([load_condition(root,run_id,condition,commit),tokenizer_for(root)])
        return bundle
    for task in T1_TASKS:
        benchmark(root,run_id,condition,task,commit,loader=loader)
    qualitative(root,run_id,condition,commit,loader=loader)


def diagnostics(root, run_id, condition, commit=lambda: None):
    from .benchmarks import tasks as benchmark_tasks
    root = Path(root)
    probe = read_json(root/"data"/"task1"/"probe_records.json")
    probe_ids=read_json(root/"data"/"task1"/"probe_ids.json")
    tok = tokenizer_for(root)
    cache_dir = root/"data"/"activation-probe"
    store = Store(root, commit)
    cache_sig = digest({"pins": read_json(root/"data"/"pins.json"), "probe": probe,
        "code": code_hash("eval1.py", "text.py")})
    if not store.done(cache_dir, cache_sig):
        model = load_qwen(root, full=True).eval()
        tasks=benchmark_tasks(root,["gsm8k_cot","hellaswag"])
        captured, current = {i: [] for i in (0, 13, 27)}, {}
        hooks = []
        for layer in captured:
            def hook(module, args, layer=layer):
                captured[layer].append(args[0][0, -1].detach().float().cpu())
            hooks.append(model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
        prompts = []
        try:
            with torch.inference_mode():
                for group, task_name in (("gsm8k", "gsm8k_cot"), ("hellaswag", "hellaswag")):
                    for doc_id in probe_ids[group]:
                        doc=tasks[task_name].eval_docs[doc_id]
                        text = tasks[task_name].doc_to_text(doc)
                        ids = chat_ids(tok, [{"role": "user", "content": text}], True) if group == "gsm8k" else tok.encode(text, add_special_tokens=False)
                        model(torch.tensor([ids], device="cuda"), use_cache=False)
                        prompts.append({"group": group, "prompt": text, "input_ids": ids})
        finally:
            for hook in hooks:
                hook.remove()
        cache_dir.mkdir(parents=True, exist_ok=True)
        with atomic_path(cache_dir/"z.pt") as tmp:
            torch.save({i: torch.stack(z) for i, z in captured.items()}, tmp)
        write_json(cache_dir/"prompts.json", prompts)
        store.finish(cache_dir, cache_sig, [cache_dir/"z.pt", cache_dir/"prompts.json"])
        del model
        torch.cuda.empty_cache()
    if condition == "baseline":
        return
    info = condition_checkpoint(root, run_id, condition)
    d = root/"runs"/run_id/"task1-diagnostics"/condition
    sig = digest({"checkpoint": info[2], "probe": cache_sig, "code": code_hash("eval1.py", "models.py")})
    if store.done(d, sig):
        return
    baseline = load_qwen(root, full=True).eval()
    basis_names={f"model.layers.{layer}.mlp.down_proj" for layer in (0,13,27)}
    bases,_=compute_bases(baseline,("down_proj",),
        root/"data"/"bases"/read_json(root/"data"/"pins.json")["models"][QWEN],
        commit=commit,module_names=basis_names)
    z = torch.load(cache_dir/"z.pt", weights_only=True)
    state = torch.load(info[0], map_location="cpu", weights_only=False)["model"]
    method = info[1]
    rows = []
    displacement_sq = 0.0
    if not T1_METHODS[method]["rank"]:
        for name, p in unique_parameters(baseline):
            displacement_sq += float((state[name].float()-p.detach().cpu()).square().sum())
    for layer in (0, 13, 27):
        name = f"model.layers.{layer}.mlp.down_proj"
        weight = baseline.get_submodule(name).weight.detach().float()
        U,V=(x.to("cuda") for x in bases[name])
        if not T1_METHODS[method]["rank"]:
            delta = state[name+".weight"].to("cuda").float()-weight
        else:
            a, b = state[name+".A"].to("cuda"), state[name+".B"].to("cuda")
            if method == "oplora":
                a, b = a-(a@V)@V.T, b-U@(U.T@b)
            cfg = T1_METHODS[method]
            delta = cfg["alpha"]/cfg["rank"]*(b@a)
        norm_sq = float(delta.square().sum())
        rho = float((U.T@delta).square().sum())/norm_sq if norm_sq else None
        eta = float(delta.norm()/weight.norm())
        values = z[layer].to("cuda")
        ratio = ((values@delta.T).norm(dim=1)/((values@weight.T).norm(dim=1)+1e-8)).cpu().tolist()
        for i, value in enumerate(ratio):
            rows.append({"condition": condition, "layer": layer, "probe_index": i,
                         "group": "math" if i < 32 else "non_math", "rho": rho, "eta": eta, "R": value})
    write_parquet(d/"diagnostics.parquet", rows)
    write_json(d/"displacement.json", {"l2_displacement": displacement_sq**.5 if not T1_METHODS[method]["rank"] else None})
    store.finish(d, sig, [d/"diagnostics.parquet", d/"displacement.json"])

