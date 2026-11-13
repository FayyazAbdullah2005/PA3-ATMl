"""One short GPU session, cached per subcheck. These are NOT assignment results."""
from __future__ import annotations
import gc
from pathlib import Path
import time
import numpy as np
import torch
from .config import *
from .io import *
from .models import *
from .text import *
from .train import seed_all,token_losses,optimizer_groups,collate_text


def smoke_signature(root):
    return digest({"pins":read_json(Path(root)/"data"/"pins.json"),
        "code":code_hash("models.py","text.py","train.py","vision.py","eval1.py","smoke.py"),
        "versions":environment()["packages"]})


def require_passed(root):
    d=Path(root)/"checks"/"gpu"
    if not Store(root).done(d,smoke_signature(root)):
        raise RuntimeError("Run the short quick-check command first. Full GPU experiments are gated until it passes.")
    measured=max((r.get("peak_gpu_bytes",0) for r in read_json(d/"summary.json")["task1"]),default=0)
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory<measured+256*1024**2:
        raise RuntimeError("This GPU is smaller than the measured preflight memory requirement. Use --gpu A100-40GB.")


def single_update(model,batch,forward,lr=2e-5,references=None,l2=0.0):
    model.train()
    opt=torch.optim.AdamW(optimizer_groups(model,lr),betas=(.9,.999),eps=1e-8)
    scaler=torch.amp.GradScaler("cuda",init_scale=256)
    for attempt in range(8):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.float16):
            sums,counts=forward(batch)
            loss=sums.sum()/counts.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Quick-check produced non-finite loss")
        scaler.scale(loss).backward()
        if l2:
            with torch.no_grad():
                for name,p in unique_parameters(model):
                    p.grad.add_(p-references[name],alpha=l2*scaler.get_scale())
        scaler.unscale_(opt)
        finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        if finite:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
        old_scale=scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if finite:
            break
    else:
        raise FloatingPointError("Quick-check overflow persisted after eight retries")
    result={"loss":float(loss.detach()),"overflow_retries":attempt,"scale":scaler.get_scale(),
        "peak_gpu_bytes":torch.cuda.max_memory_allocated(),
        "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad)}
    del opt
    return result


def quick_check(root,commit=lambda:None):
    from PIL import Image
    from transformers import SiglipVisionModel,AutoImageProcessor
    from .vision import VLM,vision_batch
    from .task2 import generate_batch,reference_nll
    root=Path(root)
    sig=smoke_signature(root)
    d=root/"checks"/"gpu"
    store=Store(root,commit)
    if store.done(d,sig):
        print("GPU quick-check already passed with this implementation/environment; no work repeated.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Quick-check requires a CUDA GPU")
    seed_all(SEED)
    tokenizer=tokenizer_for(root)
    simple=supervised_example(tokenizer,"What is 2 + 2?","4")
    results=[]
    for method,cfg in T1_METHODS.items():
        unit=d/method
        usig=digest([sig,method])
        if store.done(unit,usig):
            results.append(read_json(unit/"result.json"))
            continue
        begin=time.perf_counter()
        seed_all(SEED)
        model=load_qwen(root,full=not cfg["rank"],training=True)
        reference=None
        if method=="l2sp":
            reference={n:p.detach().clone() for n,p in unique_parameters(model)}
            # One max-length microbatch validates the largest Full-FT memory path.
            ids=(simple["input_ids"]*math_ceil(2048/len(simple["input_ids"])))[:2048]
            example={"input_ids":ids,"labels":[-100]*1024+ids[1024:]}
        else:
            example=simple
        if cfg["rank"]:
            model.requires_grad_(False)
            bases=None
            only=None
            if method=="oplora":
                # Projection GPU compatibility needs one real matrix; the full
                # 196-matrix SVD is useful production preprocessing, not a smoke test.
                name="model.layers.0.self_attn.q_proj"
                weight=model.get_submodule(name).weight.float()
                u,_,vh=torch.linalg.svd(weight,full_matrices=False)
                bases={name:(u[:,:128].contiguous(),vh[:128].T.contiguous())}
                only={name}
                del weight,u,vh
            install_lora(model,TARGETS,cfg["rank"],cfg["alpha"],bases=bases,module_names=only)
            model.enable_input_require_grads()
        batch=collate_text([example,example],tokenizer.pad_token_id,device="cuda")
        def forward(value):
            return token_losses(model(input_ids=value["input_ids"],attention_mask=value["attention_mask"],use_cache=False).logits,value["labels"])
        torch.cuda.reset_peak_memory_stats()
        result=single_update(model,batch,forward,cfg["lr"],reference,cfg["l2"])
        if method=="oplora":
            module=model.get_submodule("model.layers.0.self_attn.q_proj")
            delta=module.delta().detach()
            leakage=float((module.U.T@delta).norm()/delta.norm().clamp(min=1e-12))
            if leakage>1e-4:
                raise RuntimeError(f"OPLoRA GPU projection check failed: {leakage}")
            result["relative_output_leakage"]=leakage
        torch.cuda.synchronize()
        result.update(check=method,seconds=time.perf_counter()-begin,gpu=torch.cuda.get_device_name(),synthetic=True)
        write_json(unit/"result.json",result)
        store.finish(unit,usig,[unit/"result.json"])
        results.append(result)
        del model,batch,reference
        # forward closes over model; removing the local function releases it too.
        del forward
        gc.collect()
        torch.cuda.empty_cache()
    # One frozen-encoder forward, then remove the encoder before VLM checks.
    feature_unit=d/"siglip"
    fsig=digest([sig,"siglip"])
    if not store.done(feature_unit,fsig):
        begin=time.perf_counter()
        pins=read_json(root/"data"/"pins.json")
        encoder=SiglipVisionModel.from_pretrained(SIGLIP,revision=pins["models"][SIGLIP],
            torch_dtype=torch.float16,attn_implementation="sdpa").to("cuda").eval()
        processor=AutoImageProcessor.from_pretrained(SIGLIP,revision=pins["models"][SIGLIP])
        pixels=processor(images=[Image.new("RGB",(224,224),(100,160,200))],return_tensors="pt")["pixel_values"].to("cuda",torch.float16)
        with torch.inference_mode():
            features=encoder(pixel_values=pixels).last_hidden_state.float().cpu()
        if features.shape!=(1,196,768) or not torch.isfinite(features).all():
            raise ValueError("SigLIP GPU feature contract failed")
        with atomic_path(feature_unit/"features.pt") as tmp:
            torch.save(features,tmp)
        write_json(feature_unit/"result.json",{"check":"siglip","seconds":time.perf_counter()-begin,"shape":list(features.shape)})
        store.finish(feature_unit,fsig,[feature_unit/"features.pt",feature_unit/"result.json"])
        del encoder,pixels,processor
        gc.collect()
        torch.cuda.empty_cache()
    features=torch.load(feature_unit/"features.pt",weights_only=True)
    for tokens in (196,49):
        unit=d/f"vlm-{tokens}"
        usig=digest([sig,"vlm",tokens])
        if store.done(unit,usig):
            continue
        begin=time.perf_counter()
        seed_all(SEED)
        model=VLM(load_qwen(root,training=True))
        class InMemoryFeatures:
            def __init__(self):
                self.tokens=tokens
            def get(self,image_id):
                x=features[0].clone()
                return x if tokens==196 else x.reshape(7,2,7,2,768).mean((1,3)).reshape(49,768)
        memory=InMemoryFeatures()
        examples=[{"image_id":0,"type":"caption","question":"Describe this image.","answer":"A blue image."}]*2
        model.calibrate(examples,tokenizer,memory)
        phases=[]
        for stage in range(3):
            model.set_stage(stage)
            batch=vision_batch(examples,tokenizer,memory)
            def forward(value):
                return token_losses(model(value).logits,value["labels"].to("cuda"))
            phases.append(single_update(model,batch,forward,lr=1e-4))
            if not any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.connector.parameters()):
                raise RuntimeError("Connector did not receive gradients")
            if any(p.grad is not None for n,p in model.lm.named_parameters() if not n.endswith((".A",".B"))):
                raise RuntimeError("Frozen backbone received gradients")
        model.eval()
        generated=generate_batch(model,tokenizer,vision_batch(examples,tokenizer,memory,generation=True),2)
        omitted=generate_batch(model,tokenizer,vision_batch(examples,tokenizer,memory,generation=True,no_image=True),2)
        result={"check":f"vlm-{tokens}","seconds":time.perf_counter()-begin,"stages":phases,
                "generated":generated,"no_image":omitted,"synthetic":True}
        write_json(unit/"result.json",result)
        store.finish(unit,usig,[unit/"result.json"])
        del model,batch,forward
        gc.collect()
        torch.cuda.empty_cache()
    unit=d/"live"
    usig=digest([sig,"live"])
    if not store.done(unit,usig):
        model=load_qwen(root,name=LIVE_MODEL).eval()
        tok=tokenizer_for(root,LIVE_MODEL)
        result=generate_text(model,tok,[{"role":"system","content":"You are a helpful assistant."},
            {"role":"user","content":"What is 2+2? Give the answer in \\boxed{}."}],2,
            thinking=None,seed=SEED,sampling=True)
        result["synthetic"]=True
        write_json(unit/"result.json",result)
        store.finish(unit,usig,[unit/"result.json"])
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(d/"summary.json",{"passed":True,"gpu":torch.cuda.get_device_name(),"signature":sig,
        "notice":"Synthetic compatibility checks only; production recipes and complete evaluations are separate.",
        "task1":results})
    store.finish(d,sig,[d/"summary.json"])
    print("Short GPU compatibility checks passed. Full experiments are now enabled.",flush=True)


def math_ceil(value):
    import math
    return math.ceil(value)

