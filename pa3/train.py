"""Explicit accumulation, AMP overflow retry, and verified rolling checkpoints."""
from __future__ import annotations
import copy
from json import JSONDecodeError
import math
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from torch.nn import functional as F
from .io import (Store, atomic_path, append_event, digest, environment, read_json,
                 sha256, write_json)
from .models import trainable_state, load_trainable, unique_parameters


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def cpu_copy(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_copy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_copy(v) for v in obj)
    return copy.deepcopy(obj)


def schedule_factor(update, total, warmup, floor=0.0):
    """update is 1-based, LR is assigned before the update."""
    if not 1 <= update <= total:
        raise ValueError((update, total))
    if warmup and update <= warmup:
        return update / warmup
    progress = (update-warmup) / max(1, total-warmup)
    return floor + (1-floor)*.5*(1+math.cos(math.pi*progress))


def optimizer_groups(model, lr, weight_decay=.01):
    groups = []
    for decay in (True, False):
        params = [p for n, p in unique_parameters(model) if p.requires_grad
                  and (p.ndim >= 2 and not n.endswith("bias")) == decay]
        if params:
            groups.append({"params": params, "lr": lr,
                           "weight_decay": weight_decay if decay else 0.0, "base_lr": lr})
    return groups


def token_losses(logits, labels):
    """Returns per-example CE sums and counts, with causal shift exactly once."""
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"Label/logit shape mismatch: {labels.shape}, {logits.shape}")
    sums, counts = [], []
    for i in range(labels.shape[0]):
        target = labels[i, 1:]
        mask = target != -100
        count = int(mask.sum())
        if count == 0:
            raise ValueError("Example has no supervised response tokens")
        selected = logits[i, :-1][mask]
        truth = target[mask]
        # Checkpoint CE chunks to avoid retaining a full FP32 vocab softmax.
        parts = []
        for start in range(0, count, 128):
            x, y = selected[start:start+128], truth[start:start+128]
            def ce(a, b):
                return F.cross_entropy(a.float(), b, reduction="sum")
            if torch.is_grad_enabled() and x.requires_grad:
                from torch.utils.checkpoint import checkpoint
                parts.append(checkpoint(ce, x, y, use_reentrant=False))
            else:
                parts.append(ce(x, y))
        sums.append(torch.stack(parts).sum())
        counts.append(count)
    return torch.stack(sums), torch.tensor(counts, device=logits.device, dtype=torch.float32)


def collate_text(examples, pad_id, device="cpu", left=False):
    length = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), length), pad_id, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for i, example in enumerate(examples):
        n = len(example["input_ids"])
        start = length-n if left else 0
        ids[i, start:start+n] = torch.tensor(example["input_ids"], device=device)
        labels[i, start:start+n] = torch.tensor(example.get("labels", [-100]*n), device=device)
        mask[i, start:start+n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


class Checkpoints:
    def __init__(self, directory, signature, commit=lambda: None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.signature, self.commit = signature, commit

    def save(self, state, milestone=False):
        step = state["step"]
        name = f"step-{step:06d}" if milestone else f"recovery-{(step//25)%2}"
        target = self.directory / f"{name}.pt"
        with atomic_path(target) as tmp:
            torch.save(state, tmp)
        import hashlib
        model_digest=hashlib.sha256()
        for key,value in sorted(state["model"].items()):
            model_digest.update(key.encode("utf-8"))
            model_digest.update(str(value.dtype).encode("ascii"))
            model_digest.update(str(tuple(value.shape)).encode("ascii"))
            model_digest.update(memoryview(value.contiguous().reshape(-1).view(torch.uint8).numpy()).cast("B"))
        write_json(target.with_suffix(".json"), {"step": step, "signature": self.signature,
                   "sha256": sha256(target), "file": target.name,
                   "model_sha256":model_digest.hexdigest(),
                   "bytes": target.stat().st_size, "saved_unix": time.time()})
        self.commit()
        return target

    def latest(self, step=None):
        candidates = []
        for meta in self.directory.glob("*.json"):
            try:
                value = read_json(meta)
                if value["signature"] != self.signature:
                    raise ValueError("Checkpoint signature differs; existing data preserved")
                if step is None or value["step"] == step:
                    candidates.append(value)
            except (KeyError,OSError,JSONDecodeError,TypeError):
                continue
        candidates.sort(key=lambda x: (x["step"], x["saved_unix"]), reverse=True)
        for value in candidates:
            path = self.directory / value["file"]
            if path.is_file() and sha256(path) == value["sha256"]:
                return path, value
            append_event(self.directory / "recovery_events.jsonl", {"invalid_checkpoint": str(path)})
        if step is not None:
            raise FileNotFoundError(f"No verified checkpoint for update {step} in {self.directory}")
        return None


class Trainer:
    def __init__(self, model, optimizer, directory, configuration, commit=lambda: None,
                 full_state=False, state_names=None):
        self.model, self.optimizer = model, optimizer
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.configuration = configuration
        self.signature = digest(configuration)
        manifest = self.directory / "run.json"
        if manifest.exists() and read_json(manifest)["signature"] != self.signature:
            raise RuntimeError(f"Run settings or implementation changed: {self.directory}; use a new run ID")
        if not manifest.exists():
            write_json(manifest, {"signature": self.signature, "configuration": configuration,
                                 "environment": environment()})
        self.checkpoints = Checkpoints(self.directory / "checkpoints", self.signature, commit)
        self.scaler = torch.amp.GradScaler("cuda", enabled=next(model.parameters()).is_cuda,
                                           init_scale=256.0)
        self.step, self.elapsed, self.full_state = 0, 0.0, full_state
        self.extra = {}
        self.state_names = set(state_names) if state_names is not None else None
        self.stop_requested = False
        self.commit = commit

    def restore(self):
        latest = self.checkpoints.latest()
        if latest is None:
            return False
        # Only load our own checksum-verified checkpoint, never an untrusted pickle.
        state = torch.load(latest[0], map_location="cpu", weights_only=False)
        if state["signature"]!=self.signature:
            raise RuntimeError("Checkpoint payload/configuration signature mismatch")
        if self.full_state:
            self.model.load_state_dict(state["model"], strict=True)
        elif self.state_names is not None:
            if set(state["model"]) != self.state_names:
                raise ValueError("Checkpoint parameter set differs from the configured connector/adapters")
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if name in self.state_names:
                        parameter.copy_(state["model"][name].to(parameter.device, parameter.dtype))
        else:
            load_trainable(self.model, state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.step, self.elapsed, self.extra = state["step"], state["elapsed"], state["extra"]
        restore_rng(state["rng"])
        return True

    def save(self, milestone=False, safe_rng=None):
        start=time.perf_counter()
        state = {"step": self.step, "elapsed": self.elapsed,
                 "model": cpu_copy(self.model.state_dict()) if self.full_state else
                    ({n: p.detach().cpu().clone() for n,p in self.model.named_parameters() if n in self.state_names}
                     if self.state_names is not None else trainable_state(self.model)),
                 "optimizer": cpu_copy(self.optimizer.state_dict()),
                 "scaler": self.scaler.state_dict(), "rng": safe_rng or rng_state(),
                 "extra": self.extra, "signature": self.signature,
                 "scheduler": self.extra.get("schedule", {})}
        path = self.checkpoints.save(state, milestone)
        append_event(self.directory / "checkpoint_times.jsonl", {"step": self.step,
                     "seconds": time.perf_counter()-start, "bytes": path.stat().st_size})
        return path

    def run(self, total_steps, batch_fn, forward_fn, lr_fn, weighting="tokens",
            references=None, l2=0.0, milestones=(), callback=None, save_every=25):
        if save_every < 1:
            raise ValueError("save_every must be positive")
        device = next(self.model.parameters()).device
        hardware=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU"
        segments=self.extra.setdefault("hardware_segments",[])
        if not segments or segments[-1]["hardware"]!=hardware:
            segments.append({"hardware":hardware,"start_update":self.step+1})
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        old_handlers = {}
        def request_stop(*_):
            self.stop_requested = True
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_handlers[sig] = signal.signal(sig, request_stop)
        except ValueError:
            pass  # A test caller may run in a worker thread.
        self.model.train()
        safe_to_save = True
        safe_rng = rng_state()
        try:
            while self.step < total_steps:
                index = self.step
                safe_rng = rng_state()
                batch_begin=time.perf_counter()
                batches = batch_fn(index)
                self.extra["data_seconds"]=self.extra.get("data_seconds",0)+time.perf_counter()-batch_begin
                if not batches:
                    raise ValueError("Empty effective batch")
                total_tokens = sum(int((b["labels"][:, 1:] != -100).sum()) for b in batches)
                total_examples = sum(b["labels"].shape[0] for b in batches)
                if total_tokens == 0:
                    raise ValueError("No supervised tokens in effective batch")
                schedule = lr_fn(index)
                for group in self.optimizer.param_groups:
                    role = group.get("role", "default")
                    group["lr"] = schedule[role] if isinstance(schedule, dict) else schedule
                self.extra["schedule"] = {"update": index+1,
                    "lrs": [g["lr"] for g in self.optimizer.param_groups]}
                begin = time.perf_counter()
                for attempt in range(8):
                    restore_rng(safe_rng)
                    self.optimizer.zero_grad(set_to_none=True)
                    sft = 0.0
                    safe_to_save = True
                    for batch in batches:
                        with torch.autocast(device_type=device.type, dtype=torch.float16,
                                            enabled=device.type == "cuda"):
                            sums, counts = forward_fn(batch)
                            loss = sums.sum()/total_tokens if weighting == "tokens" else (sums/counts).sum()/total_examples
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Non-finite loss at update {index+1}")
                        self.scaler.scale(loss).backward()
                        sft += float(loss.detach())
                    # Add the exact analytic L2-SP gradient while grads are scaled,
                    # so GradScaler also checks the penalty contribution for overflow.
                    omega = 0.0
                    if l2:
                        if references is None:
                            raise ValueError("L2-SP requires fixed starting parameters")
                        scale = self.scaler.get_scale()
                        with torch.no_grad():
                            for name, parameter in unique_parameters(self.model):
                                diff = parameter.float() - references[name]
                                omega += .5*float(diff.square().sum())
                                if parameter.grad is None:
                                    raise RuntimeError(f"Missing gradient: {name}")
                                parameter.grad.add_(diff, alpha=l2*scale)
                    self.scaler.unscale_(self.optimizer)
                    parameters = [p for p in self.model.parameters() if p.requires_grad]
                    grads = [p.grad for p in parameters if p.grad is not None]
                    if not grads:
                        raise RuntimeError("No trainable parameter received a gradient")
                    finite = all(bool(torch.isfinite(g).all()) for g in grads)
                    if not finite:
                        # step() records the skip; update() reduces the scale.
                        if not self.scaler.is_enabled():
                            raise FloatingPointError("Non-finite FP32 gradients")
                        previous = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        append_event(self.directory / "overflow.jsonl", {"step": index+1,
                            "attempt": attempt+1, "old_scale": previous, "new_scale": self.scaler.get_scale()})
                        continue
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    safe_to_save = False  # A failed optimizer kernel may mutate only some weights.
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    self.step += 1
                    self.elapsed += time.perf_counter()-begin
                    self.extra["examples_seen"] = self.extra.get("examples_seen", 0)+total_examples
                    if "pending_mixture" in self.extra:
                        mixture=self.extra.setdefault("mixture_counts",{})
                        for key,value in self.extra.pop("pending_mixture").items():
                            mixture[key]=mixture.get(key,0)+value
                    safe_rng = rng_state()
                    safe_to_save = True
                    log = {"step": self.step, "sft_loss": sft, "omega": omega,
                           "hardware":hardware,
                           "weighted_penalty": l2*omega, "total_loss": sft+l2*omega,
                           "learning_rates": [g["lr"] for g in self.optimizer.param_groups],
                           "gradient_norm": float(norm), "examples": total_examples,
                           "response_tokens": total_tokens, "training_seconds": self.elapsed,
                           "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}
                    append_event(self.directory / "training.jsonl", log)
                    if self.step % 10 == 0 or self.step == total_steps:
                        print(f"update {self.step}/{total_steps}  loss={sft:.5f}  elapsed={self.elapsed:.1f}s", flush=True)
                    break
                else:
                    raise FloatingPointError("AMP overflow persisted for eight retries; latest recovery checkpoint preserved")
                if self.step % save_every == 0 or self.step in milestones or self.step == total_steps:
                    self.save(milestone=self.step in milestones or self.step == total_steps)
                if callback:
                    before = rng_state()
                    callback(self)
                    restore_rng(before)
                    self.model.train()
                if self.stop_requested:
                    self.save(safe_rng=safe_rng)
                    raise InterruptedError("Stopped cleanly after saving a complete optimizer update; rerun the same command")
        except BaseException:
            if safe_to_save:
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.save(safe_rng=safe_rng)
                except BaseException as error:
                    print(f"Emergency save failed ({error}); previous verified checkpoint remains available", flush=True)
            raise
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        return self.step

