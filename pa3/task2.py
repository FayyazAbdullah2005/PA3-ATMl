"""Three-stage VLM curriculum, image-control evaluation, and pair bootstraps."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import math
import time
import numpy as np
import torch
from .config import *
from .data2 import draw_grounding
from .io import *
from .models import LoRALinear
from .scoring import vqa_normalize, vqa_soft, bootstrap_difference
from .text import load_qwen, tokenizer_for
from .train import *
from .vision import Features,VLM,vision_batch


def stage_position(global_step):
    start = 0
    for i,stage in enumerate(T2_STAGES):
        if global_step < start+stage["updates"]:
            return i,global_step-start
        start += stage["updates"]
    return 2,T2_STAGES[2]["updates"]


def development_steps():
    return {0,2250,3750,4500,*range(500,4501,500)}


def data_for(root,tokens):
    d = Path(root)/"data"/"task2"
    return {"pairs":read_json(d/"pairs.json"),"captions":read_json(d/"captions.json"),
            "pools":read_json(d/"grounding_pools.json"),
            "mismatched":read_json(d/"mismatched.json"),
            "contractions":read_json(d/"contractions.json"),
            "features":Features(d/"features",tokens)}


def draw_effective_batch(data,step):
    stage,local = stage_position(step)
    rng = np.random.default_rng(np.random.SeedSequence([SEED,2,step]))
    counts = Counter()
    if stage == 0:
        # Exactly two complete passes; each pass has its own fixed permutation.
        epoch,index = divmod(local*16,18000)
        order = np.random.default_rng(np.random.SeedSequence([SEED,20,epoch])).permutation(18000)
        selected = [data["captions"]["train"][int(i)] for i in order[index:index+16]]
        counts["caption"] = 16
    else:
        mixture = T2_STAGES[stage]["mixture"]
        kinds,probabilities = list(mixture),list(mixture.values())
        selected = []
        for draw in range(8):
            kind = str(rng.choice(kinds,p=probabilities))
            if kind == "caption":
                chosen = rng.integers(len(data["captions"]["train"]),size=2)
                examples = [data["captions"]["train"][int(i)] for i in chosen]
            elif kind == "natural":
                examples = data["pairs"]["natural_train"][int(rng.integers(len(data["pairs"]["natural_train"])))]["examples"]
            else:
                examples = draw_grounding(data["pools"],kind,rng,pair_id=f"train-{step}-{draw}")["examples"]
            selected.extend(examples)
            counts[kind] += len(examples)
    if len(selected) != 16:
        raise ValueError("VLM effective batch must contain exactly 16 examples")
    return selected,dict(counts)


@torch.inference_mode()
def reference_nll(model,tokenizer,features,examples,image_ids=None,no_image=False,caption=False):
    batch = vision_batch(examples,tokenizer,features,image_ids,no_image=no_image)
    labels = batch["labels"].to(next(model.parameters()).device)
    if not caption:
        labels = labels.clone()
        labels[labels == tokenizer.convert_tokens_to_ids("<|im_end|>")] = -100
    with torch.autocast("cuda",dtype=torch.float16):
        sums,counts = token_losses(model(batch).logits,labels)
    nll=sums/counts
    if not torch.isfinite(nll).all():
        raise FloatingPointError("Non-finite reference NLL; no completed evaluation record was written")
    return nll.cpu().tolist()


def dev_evaluate(model,tokenizer,data,directory,update,signature,commit=lambda:None):
    d = Path(directory)/"development"/f"step-{update:06d}"
    store = Store(directory,commit)
    sig = digest([signature,update,"development"])
    if store.done(d,sig):
        return
    model.eval()
    rows = []
    captions = data["captions"]["val"]
    for start in range(0,len(captions),8):
        examples = captions[start:start+8]
        values = reference_nll(model,tokenizer,data["features"],examples,caption=True)
        rows.extend({"suite":"caption","condition":"correct","id":str(e["caption_id"]),"nll":v}
                    for e,v in zip(examples,values))
    for suite in ("grounding_dev","natural_dev"):
        pairs = data["pairs"][suite]
        examples = [e for p in pairs for e in p["examples"]]
        partners = {str(e["question_id"]):p["examples"][1-i]["image_id"] for p in pairs for i,e in enumerate(p["examples"])}
        for condition in ("correct","swap","none"):
            for start in range(0,len(examples),8):
                chunk = examples[start:start+8]
                images = [partners[str(e["question_id"])] for e in chunk] if condition=="swap" else None
                values = reference_nll(model,tokenizer,data["features"],chunk,images,no_image=condition=="none")
                rows.extend({"suite":suite,"condition":condition,"id":str(e["question_id"]),
                             "type":e["type"],"nll":v} for e,v in zip(chunk,values))
    write_parquet(d/"examples.parquet",rows)
    summary = {f"{suite}/{condition}":float(np.mean([r["nll"] for r in rows if r["suite"]==suite and r["condition"]==condition]))
               for suite,condition in sorted({(r["suite"],r["condition"]) for r in rows})}
    for suite in ("grounding_dev","natural_dev"):
        summary[suite+"/swap_penalty"] = summary[suite+"/swap"]-summary[suite+"/correct"]
    write_json(d/"summary.json",{"update":update,"metrics":summary})
    store.finish(d,sig,[d/"examples.parquet",d/"summary.json"])


def training_configuration(root,tokens,smoke=False):
    d=Path(root)/"data"/"task2"
    if not Store(root).done(d) or not Store(root).done(d/"features"):
        raise RuntimeError("Run task2-prepare and task2-features first")
    return {"task":2,"tokens":tokens,"seed":SEED,"stages":T2_STAGES,"smoke":smoke,
            "shared":T2_SHARED,"pins":read_json(Path(root)/"data"/"pins.json"),
            "updates":3 if smoke else 4500,"data":read_json(d/"DONE.json")["signature"],
            "features":read_json(d/"features"/"DONE.json")["signature"],
            "code":code_hash("task2.py","vision.py","models.py","train.py","text.py","scoring.py")}


def train(root,tokens=196,run_id="main",smoke=False,commit=lambda:None):
    root=Path(root)
    configuration=training_configuration(root,tokens,smoke)
    d=root/"runs"/("smoke" if smoke else run_id)/"task2"/str(tokens)
    store=Store(root,commit)
    sig=digest(configuration)
    if store.done(d,sig):
        try:
            Checkpoints(d/"checkpoints",sig).latest(configuration["updates"])
            return
        except FileNotFoundError:
            print("Final VLM checkpoint is unavailable; resuming verified recovery state.",flush=True)
    seed_all(SEED)
    model=VLM(load_qwen(root,training=True))
    tokenizer=tokenizer_for(root)
    data=data_for(root,tokens)
    # Calibration is immutable and restored separately from trainable parameters.
    if (d/"calibration.json").exists():
        calibration=read_json(d/"calibration.json")
        model.visual_scale.copy_(torch.tensor(calibration["scale"],device="cuda"))
    else:
        calibration=model.calibrate(data["captions"]["train"][:16],tokenizer,data["features"])
        write_json(d/"calibration.json",calibration)
        commit()
    names={n for n,p in model.named_parameters() if n.startswith("connector.") or n.endswith((".A",".B"))}
    groups=[]
    for role in ("connector","adapter"):
        for decay in (True,False):
            parameters=[p for n,p in model.named_parameters() if n in names and
                        (n.startswith("connector."))==(role=="connector") and
                        (p.ndim>=2 and not n.endswith("bias"))==decay]
            if parameters:
                groups.append({"params":parameters,"lr":0.0,"role":role,"weight_decay":.01 if decay else 0.0})
    optimizer=torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8)
    trainer=Trainer(model,optimizer,d,configuration,commit,state_names=names)
    trainer.restore()
    model.set_stage(0 if smoke else stage_position(trainer.step)[0])
    if not smoke and trainer.step in development_steps():
        dev_evaluate(model,tokenizer,data,d,trainer.step,sig,commit)
    def batches(step):
        # Three smoke updates exercise all three stage paths, with no long dev sweep.
        actual=(0,2250,3750)[step] if smoke else step
        examples,counts=draw_effective_batch(data,actual)
        trainer.extra["pending_mixture"]=counts
        return [vision_batch(examples[i:i+8],tokenizer,data["features"]) for i in (0,8)]
    def forward(batch):
        return token_losses(model(batch).logits,batch["labels"].to("cuda"))
    def lr(step):
        actual=(0,2250,3750)[step] if smoke else step
        stage,local=stage_position(actual)
        model.set_stage(stage)
        cfg=T2_STAGES[stage]
        factor=schedule_factor(local+1,cfg["updates"],math.ceil(.05*cfg["updates"]),.1)
        return {"connector":cfg["connector_lr"]*factor,"adapter":cfg["adapter_lr"]*factor}
    def callback(t):
        mixture=t.extra.setdefault("mixture_counts",{})
        for k,v in t.extra.pop("pending_mixture",{}).items():
            mixture[k]=mixture.get(k,0)+v
        append_event(d/"mixture.jsonl",{"step":t.step,"counts":dict(mixture)})
        if smoke:
            return
        if t.step in development_steps():
            dev_evaluate(model,tokenizer,data,d,t.step,sig,commit)
        if t.step in (2250,3750,4500):
            endpoint=d/"endpoints"/str(t.step)
            with atomic_path(endpoint/"weights.pt") as tmp:
                torch.save({"parameters":{n:p.detach().cpu() for n,p in model.named_parameters() if n in names},
                            "visual_scale":float(model.visual_scale)},tmp)
            write_json(endpoint/"resources.json",{"update":t.step,"training_seconds":t.elapsed,
                "trainable_parameters":sum(p.numel() for n,p in model.named_parameters() if n in names and
                    (t.step>2250 or n.startswith("connector."))),
                "peak_gpu_bytes":torch.cuda.max_memory_allocated(),"mixture_counts":dict(mixture)})
            commit()
    if trainer.step>0 and not smoke and trainer.step in development_steps():
        callback(trainer)
    trainer.run(configuration["updates"],batches,forward,lr,weighting="examples",
                milestones=() if smoke else (2250,3750,4500),callback=callback)
    # Callback metadata is saved after it has been updated, including the final mixture.
    trainer.save(milestone=True)
    write_json(d/"completed.json",{"updates":trainer.step,"training_seconds":trainer.elapsed,
                                 "mixture_counts":trainer.extra.get("mixture_counts",{})})
    store.finish(d,sig,[d/"run.json",d/"calibration.json",d/"completed.json"])


def load_final(root,run_id,tokens):
    d=Path(root)/"runs"/run_id/"task2"/str(tokens)
    if not Store(root).done(d):
        raise RuntimeError("VLM training has not completed")
    run=read_json(d/"run.json")
    info=Checkpoints(d/"checkpoints",run["signature"]).latest(4500)
    state=torch.load(info[0],map_location="cpu",weights_only=False)
    model=VLM(load_qwen(root)).eval()
    model.set_stage(2)
    actual={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(actual)!=set(state["model"]):
        raise ValueError("VLM final-checkpoint trainable keys do not match")
    with torch.no_grad():
        for n,p in actual.items():
            p.copy_(state["model"][n].to(p.device,p.dtype))
        model.visual_scale.copy_(torch.tensor(read_json(d/"calibration.json")["scale"],device="cuda"))
    return model,info[1].get("model_sha256",info[1]["sha256"])


@torch.inference_mode()
def generate_batch(model,tokenizer,batch,max_new_tokens=16):
    with torch.autocast("cuda",dtype=torch.float16):
        embeddings,mask,_=model.embedding_inputs(batch)
        torch.cuda.synchronize()
        begin=time.perf_counter()
        # Native generation derives and updates positions from attention_mask.
        output=model.lm.generate(inputs_embeds=embeddings,attention_mask=mask,
            max_new_tokens=max_new_tokens,do_sample=False,use_cache=True,
            pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        seconds=time.perf_counter()-begin
    eos=model.lm.generation_config.eos_token_id
    stops=set(eos if isinstance(eos,list) else [eos])
    results=[]
    for tokens in output.tolist():
        end=next((i+1 for i,t in enumerate(tokens) if t in stops),len(tokens))
        tokens=tokens[:end]
        results.append({"prediction":tokenizer.decode(tokens,skip_special_tokens=True).strip(),
            "token_ids":tokens,"hit_max_new_tokens":len(tokens)==max_new_tokens and (not tokens or tokens[-1] not in stops),
            "generation_seconds":seconds/len(output)})
    return results


def evaluate(root,run_id="main",tokens=196,commit=lambda:None):
    root=Path(root)
    data=data_for(root,tokens)
    d=root/"runs"/run_id/"task2-evaluation"/str(tokens)
    run=read_json(root/"runs"/run_id/"task2"/str(tokens)/"run.json")
    info=Checkpoints(root/"runs"/run_id/"task2"/str(tokens)/"checkpoints",run["signature"]).latest(4500)
    sig=digest({"checkpoint":info[1].get("model_sha256",info[1]["sha256"]),
        "calibration":sha256(root/"runs"/run_id/"task2"/str(tokens)/"calibration.json"),
        "code":code_hash("task2.py","vision.py","scoring.py")})
    store=Store(root,commit)
    unit_store=Store(root)
    if store.done(d,sig):
        return
    model=None
    all_rows=[]
    tokenizer=tokenizer_for(root)
    for suite in ("grounding_test","natural_test"):
        pairs=data["pairs"][suite]
        examples=[]
        for pair in pairs:
            for i,e in enumerate(pair["examples"]):
                examples.append({**e,"pair_id":pair["pair_id"],
                                 "partner_image":pair["examples"][1-i]["image_id"],
                                 "partner_answer":pair["examples"][1-i]["canonical"]})
        for condition in IMAGE_CONDITIONS:
            computed_new=False
            for start in range(0,len(examples),8):
                chunk=examples[start:start+8]
                unit=d/suite/condition/f"batch-{start//8:04d}"
                usig=digest([sig,suite,condition,start])
                if not unit_store.done(unit,usig):
                    if model is None:
                        model,_=load_final(root,run_id,tokens)
                    images=([e["partner_image"] for e in chunk] if condition=="swap" else
                            [data["mismatched"][suite][str(e["question_id"])] for e in chunk] if condition=="mismatched" else None)
                    batch=vision_batch(chunk,tokenizer,data["features"],images,generation=True,no_image=condition=="none")
                    predictions=generate_batch(model,tokenizer,batch)
                    nlls=reference_nll(model,tokenizer,data["features"],chunk,images,no_image=condition=="none")
                    rows=[]
                    for e,p,nll in zip(chunk,predictions,nlls):
                        normalized=vqa_normalize(p["prediction"],data["contractions"])
                        score=(vqa_soft(p["prediction"],e["references"],data["contractions"]) if suite=="natural_test"
                               else float(normalized==e["canonical"]))
                        rows.append({**e,**p,"normalized_prediction":normalized,"score":score,
                                     "canonical_match":bool(normalized and normalized==e["canonical"]),
                                     "partner_match":bool(normalized and normalized==e["partner_answer"]),
                                     "nll":nll,"suite":suite,"condition":condition,"tokens":tokens})
                    write_parquet(unit/"examples.parquet",rows)
                    unit_store.finish(unit,usig,[unit/"examples.parquet"])
                    computed_new=True
                all_rows.extend(read_parquet(unit/"examples.parquet"))
            if computed_new:
                commit()
    write_parquet(d/"examples.parquet",all_rows)
    store.finish(d,sig,[d/"examples.parquet"])


def statistics(root,run_id="main",tokens=196,commit=lambda:None):
    """CPU-only pair statistics; the GPU worker finishes before bootstrapping."""
    root=Path(root)
    source=root/"runs"/run_id/"task2-evaluation"/str(tokens)
    store=Store(root,commit)
    if not store.done(source):
        raise RuntimeError("Run task2-eval first")
    sig=digest([read_json(source/"DONE.json")["signature"],code_hash("task2.py","scoring.py"),"pair-statistics"])
    d=source/"statistics"
    if store.done(d,sig):
        return
    all_rows=read_parquet(source/"examples.parquet")
    summary=[]
    advantages=[]
    for suite in ("grounding_test","natural_test"):
        for condition in IMAGE_CONDITIONS:
            records=[r for r in all_rows if r["suite"]==suite and r["condition"]==condition]
            for kind in ["all"]+sorted({r["type"] for r in records}):
                chosen=[r for r in records if kind=="all" or r["type"]==kind]
                grouped={pid:[r for r in chosen if r["pair_id"]==pid] for pid in sorted({r["pair_id"] for r in chosen})}
                if any(len(v)!=2 for v in grouped.values()):
                    raise ValueError("Incomplete evaluation pair")
                metric_names=("accuracy","pair_accuracy","identical_answer_rate","empty_answer_rate","partner_match_rate","reference_nll","seconds_per_answer")
                matrix=np.array([[np.mean([r["score"] for r in v]),all(r["canonical_match"] for r in v),
                    v[0]["normalized_prediction"]==v[1]["normalized_prediction"],
                    np.mean([not r["normalized_prediction"] for r in v]),
                    np.mean([r["partner_match"] for r in v]),
                    np.mean([r["nll"] for r in v]),
                    np.mean([r["generation_seconds"] for r in v])] for v in grouped.values()],dtype=float)
                indices=np.random.default_rng(SEED).integers(len(matrix),size=(4000,len(matrix)))
                boot=matrix[indices].mean(axis=1)
                row={"suite":suite,"condition":condition,"type":kind,"tokens":tokens}
                for j,name in enumerate(metric_names):
                    row[name]=float(matrix[:,j].mean())
                    row[name+"_low"]=float(np.quantile(boot[:,j],.025))
                    row[name+"_high"]=float(np.quantile(boot[:,j],.975))
                summary.append(row)
        suite_rows=[r for r in all_rows if r["suite"]==suite]
        for kind in ["all"]+sorted({r["type"] for r in suite_rows}):
            pair_ids=sorted({r["pair_id"] for r in suite_rows if kind=="all" or r["type"]==kind})
            for metric in ("score","nll"):
                values={condition:[np.mean([r[metric] for r in suite_rows if r["condition"]==condition and r["pair_id"]==pid])
                                   for pid in pair_ids] for condition in ("correct","swap")}
                a,b=(values["correct"],values["swap"]) if metric=="score" else (values["swap"],values["correct"])
                advantages.append({"suite":suite,"type":kind,"metric":metric,"tokens":tokens,**bootstrap_difference(a,b,4000)})
    write_json(d/"summary.json",summary)
    write_json(d/"correct_image_advantages.json",advantages)
    store.finish(d,sig,[d/"summary.json",d/"correct_image_advantages.json"])

