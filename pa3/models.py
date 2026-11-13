"""Plain PyTorch LoRA/OPLoRA; factors remain FP32 under FP16 autocast."""
from __future__ import annotations
import math
from pathlib import Path
import time
import torch
from torch import nn
from torch.nn import functional as F
from .config import SEED
from .io import atomic_path, sha256, write_json, read_json


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout=.05,
                 bases=None, generator=None):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.enabled = True
        self.dropout = nn.Dropout(dropout)
        # Always generate on CPU with an independent RNG, unaffected by SVD or data.
        initial = torch.empty(rank, base.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(initial, a=math.sqrt(5), generator=generator)
        self.A = nn.Parameter(initial.to(base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        self.register_buffer("U", bases[0].to(base.weight.device, torch.float32) if bases else None)
        self.register_buffer("V", bases[1].to(base.weight.device, torch.float32) if bases else None)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def factors(self):
        with torch.autocast(device_type=self.A.device.type, enabled=False):
            a, b = self.A.float(), self.B.float()
            if self.U is not None:
                b = b - self.U @ (self.U.T @ b)
                a = a - (a @ self.V) @ self.V.T
        return a, b

    def delta(self):
        a, b = self.factors()
        return (self.alpha / self.rank) * (b @ a)

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        a, b = self.factors()
        return self.base(x) + (self.alpha / self.rank) * F.linear(F.linear(self.dropout(x), a), b)


def install_lora(model, targets, rank=16, alpha=32, bases=None, seed=SEED, module_names=None):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    replaced = []
    for name, module in list(model.named_modules()):
        if module_names is not None and name not in module_names:
            continue
        if name.rsplit(".", 1)[-1] in targets and isinstance(module, nn.Linear):
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            basis = bases[name] if bases is not None else None
            setattr(parent, attr, LoRALinear(module, rank, alpha, bases=basis, generator=generator))
            replaced.append(name)
    if not replaced:
        raise ValueError("No LoRA target modules matched")
    return replaced


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state):
    actual = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(actual) != set(state):
        raise ValueError(f"Trainable-state keys differ: missing={set(actual)-set(state)}, extra={set(state)-set(actual)}")
    with torch.no_grad():
        for name, parameter in actual.items():
            if parameter.shape != state[name].shape:
                raise ValueError(f"Shape mismatch: {name}")
            parameter.copy_(state[name].to(parameter.device, parameter.dtype))


@torch.no_grad()
def merge_lora(model):
    """Merge into the original module. FP32 addition followed by its original dtype."""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            dtype = module.base.weight.dtype
            updated = module.base.weight.float() + module.delta()
            module.base.weight.copy_(updated.to(dtype))
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, module.base)
    return model


def unique_parameters(model):
    # named_parameters(remove_duplicate=True) is the PyTorch default; be explicit.
    return list(model.named_parameters(remove_duplicate=True))


@torch.no_grad()
def compute_bases(model, targets, path: Path, rank=128, commit=lambda: None, module_names=None):
    """Persist each SVD separately: an interruption never repeats completed SVDs."""
    from .io import Store, digest
    store = Store(path)
    result, timings, computed = {}, [], 0
    for name, module in model.named_modules():
        if module_names is not None and name not in module_names:
            continue
        if name.rsplit(".", 1)[-1] not in targets or not isinstance(module, nn.Linear):
            continue
        d = Path(path) / name
        sig = digest({"name": name, "shape": list(module.weight.shape), "rank": rank})
        if store.done(d, sig):
            data = torch.load(d / "basis.pt", map_location="cpu", weights_only=True)
        else:
            begin = time.perf_counter()
            # Economy SVD is exact; do not instantiate dense projection matrices.
            u, _, vh = torch.linalg.svd(module.weight.detach().float(), full_matrices=False)
            data = {"U": u[:, :rank].contiguous().cpu(), "V": vh[:rank].T.contiguous().cpu()}
            d.mkdir(parents=True, exist_ok=True)
            with atomic_path(d / "basis.pt") as tmp:
                torch.save(data, tmp)
            write_json(d / "timing.json", {"seconds": time.perf_counter()-begin,
                       "bytes": sum(x.numel()*x.element_size() for x in data.values())})
            store.finish(d, sig, [d / "basis.pt", d / "timing.json"])
            computed += 1
            if computed % 28 == 0:
                commit()
            del u, vh
        result[name] = (data["U"], data["V"])
        timings.append(read_json(d / "timing.json"))
    if computed % 28 != 0:
        commit()
    return result, timings

