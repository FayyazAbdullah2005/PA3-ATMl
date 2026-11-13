"""CPU-first test-time compute experiments, with prefix-local decisions."""
from __future__ import annotations
from collections import defaultdict
import importlib.metadata
from pathlib import Path
import shutil
import warnings
import numpy as np
from .config import *
from .io import *
from .scoring import *

KEYS=("dataset","sample_id","rollout_id")
ATTENTION=("img_attn_total","prompt_text_attn_total","generated_text_attn_total")


def key_index(table):
    rows=table.select(KEYS).to_pylist()
    index={tuple(row[k] for k in KEYS):i for i,row in enumerate(rows)}
    if len(index)!=len(rows):
        raise ValueError("Duplicate rollout join key")
    return index


def validate_tables(generations,probabilities,attention=None):
    import pyarrow.compute as pc
    gi,pi=key_index(generations),key_index(probabilities)
    if gi.keys()!=pi.keys():
        raise ValueError("Generation/probability join keys do not match exactly")
    if attention is not None and gi.keys()!=key_index(attention).keys():
        raise ValueError("Generation/attention join keys do not match exactly")
    lengths={key:len(generations["generated_token_ids"][i].as_py()) for key,i in gi.items()}
    for key,i in gi.items():
        if lengths[key]<1 or lengths[key]!=generations["n_generated_text_tokens"][i].as_py():
            raise ValueError(f"Invalid retained length: {key}")
    for table,index,fields in [(probabilities,pi,("logp","kl_u_p"))]+(
            [(attention,key_index(attention),ATTENTION)] if attention is not None else []):
        for field in fields:
            list_lengths=pc.list_value_length(table[field]).to_pylist()
            if any(list_lengths[i]!=lengths[key] for key,i in index.items()):
                raise ValueError(f"Token-list alignment failed for {field}")
            flat=pc.list_flatten(table[field])
            if pc.any(pc.is_null(flat)).as_py() or pc.any(pc.invert(pc.is_finite(flat))).as_py():
                raise ValueError(f"Non-finite token values: {field}")
    if attention is not None:
        ai=key_index(attention)
        for key,i in ai.items():
            sequence=attention["n_generated_text_tokens"][i].as_py()
            if sequence!=list(range(lengths[key])):
                raise ValueError("Attention preceding-token counts are not aligned")
    groups=defaultdict(set)
    for dataset,sample,rollout in gi:
        if not isinstance(sample,str):
            raise TypeError("Question IDs must remain strings")
        groups[(dataset,sample)].add(rollout)
    if any(v!=set(range(32)) for v in groups.values()):
        raise ValueError("Every question must have exactly rollout IDs 0 through 31")
    if len(groups)!=1000:
        raise ValueError(f"Expected 1000 questions, found {len(groups)}")
    return gi,pi,key_index(attention) if attention is not None else None


def prepare(root,source=None,commit=lambda:None):
    import pyarrow.parquet as pq
    root=Path(root)
    d=root/"data"/"task3"
    store=Store(root,commit)
    raw=Path(source) if source else root/"data"/"rollouts"
    expected=[raw/"README.MD",raw/"qwen25-math"/"generations.parquet",raw/"qwen25-math"/"probability_metrics.parquet",
              raw/"qwen25vl-mathvista"/"generations.parquet",raw/"qwen25vl-mathvista"/"probability_metrics.parquet",
              raw/"qwen25vl-mathvista"/"attention_metrics.parquet"]
    if not all(p.exists() for p in expected):
        if source:
            raise FileNotFoundError(f"Supplied rollout README.MD not found at {raw}")
        import gdown
        raw.mkdir(parents=True,exist_ok=True)
        # gdown's folder downloader skips complete existing files and can resume.
        gdown.download_folder(ROLLOUT_FOLDER,output=str(raw),quiet=False,remaining_ok=True,resume=True,use_cookies=False)
        if not all(p.exists() for p in expected):
            raise FileNotFoundError("Supplied rollout download is incomplete; rerun the same preparation command")
    readme=(raw/"README.MD").read_text(encoding="utf-8")
    if LIVE_REVISION not in readme or MATH_REVISION not in readme:
        raise ValueError("The supplied README has different pinned identities; refusing an unverified live-model revision")
    metadata_path=download("https://raw.githubusercontent.com/lupantech/MathVista/main/data/testmini.json",d/"mathvista_metadata.json")
    metadata=read_json(metadata_path)
    sig=digest({"readme":sha256(raw/"README.MD"),"metadata":sha256(metadata_path),
        "scorer":importlib.metadata.version("math-verify"),"code":code_hash("task3.py","scoring.py")})
    if store.done(d,sig):
        return
    unit_store=Store(root)
    manifests={}
    for dataset_number,(name,folder) in enumerate((("math","qwen25-math"),("mathvista","qwen25vl-mathvista"))):
        directory=raw/folder
        paths=[directory/"generations.parquet",directory/"probability_metrics.parquet"]
        if name=="mathvista":
            paths.append(directory/"attention_metrics.parquet")
        tables=[pq.read_table(path,memory_map=True) for path in paths]
        indexes=validate_tables(*tables)
        ids=sorted({key[1] for key in indexes[0]})
        permutation=np.random.default_rng(SEED).permutation(len(ids))
        shuffled=[ids[int(i)] for i in permutation]
        split={"development":shuffled[:200],"evaluation":shuffled[200:]}
        orders={qid:[np.random.default_rng(np.random.SeedSequence([SEED,dataset_number,i,order])).permutation(32).tolist()
                     for order in range(10)] for i,qid in enumerate(ids)}
        manifests[name]={"splits":split,"orders":orders,"question_ids":ids,
            "live_ids":split["evaluation"][:50] if name=="math" else [],
            "files":{p.name:sha256(p) for p in paths}}
        keys_by_question=defaultdict(list)
        for key in indexes[0]:
            keys_by_question[key[1]].append(key)
        prepared_new=0
        for qid in ids:
            unit=d/name/digest(qid)[:20]
            usig=digest([sig,name,qid,manifests[name]["files"]])
            if unit_store.done(unit,usig):
                continue
            keys=sorted(keys_by_question[qid],key=lambda key:key[2])
            parts=[table.take([index[key] for key in keys]).to_pylist() for table,index in zip(tables,indexes) if table is not None]
            rows=[]
            for i in range(32):
                row=dict(parts[0][i])
                for part in parts[1:]:
                    for field,value in part[i].items():
                        if field in KEYS:
                            continue
                        row["preceding_generated_counts" if field=="n_generated_text_tokens" else field]=value
                answer=row["extracted_answer"]
                if name=="math":
                    normalized=None if answer is None or not str(answer).strip() else str(answer).strip()
                    correct=math_equivalent(normalized,str(row["gold_answer"]))
                else:
                    meta=metadata[qid]
                    normalized=mathvista_normalize(answer,meta)
                    correct=normalized is not None and normalized==mathvista_normalize(str(row["gold_answer"]),meta)
                row.update(normalized_answer=normalized,correct=bool(correct),retained_tokens=int(row["n_generated_text_tokens"]))
                rows.append(row)
            if len({r["question_text"] for r in rows})!=1 or len({str(r["gold_answer"]) for r in rows})!=1:
                raise ValueError(f"Inconsistent problem/reference inside question {qid}")
            write_parquet(unit/"rollouts.parquet",rows)
            unit_store.finish(unit,usig,[unit/"rollouts.parquet"])
            prepared_new+=1
            if prepared_new%50==0:
                commit()
        if prepared_new%50!=0:
            commit()
        del tables
    write_json(d/"manifest.json",manifests)
    write_json(d/"scorer.json",{"math_verify":importlib.metadata.version("math-verify"),
        "source_readme_sha256":sha256(raw/"README.MD"),"mathvista_metadata_sha256":sha256(metadata_path),
        "seed":SEED,"no_answer":"one missing-answer group, always incorrect"})
    store.finish(d,sig,[d/"manifest.json",d/"scorer.json",metadata_path])
    # Download weights on CPU ahead of the live GPU job.
    from .text import ensure_pins,cache_model
    ensure_pins(root,commit)
    cache_model(root,LIVE_MODEL,commit)


def public_rollout(row):
    fields=("normalized_answer","retained_tokens","logp","kl_u_p",*ATTENTION)
    return {key:row[key] for key in fields if key in row}


def prefix_state(rollouts,dataset):
    """Only visible, gold-free candidates enter the decision engine."""
    if any("correct" in r or "gold_answer" in r for r in rollouts):
        raise ValueError("Correctness/reference data must not enter selection or stopping")
    answers=[r["normalized_answer"] for r in rollouts]
    equivalent=math_equivalent if dataset=="math" else lambda a,b:a==b
    groups,representatives=group_answers(answers,equivalent)
    scores=mean_scores(rollouts,vision=dataset=="mathvista")
    methods=["first","sc","logp","certainty","certainty_rwc"]
    if dataset=="mathvista":
        methods += ["vision","vision_rwc","visconf","visconf_rwc"]
    selected={method:select(groups,scores,method) for method in methods}
    sc_group,_,unique,totals=vote(groups)
    quality="certainty" if dataset=="math" else "visconf"
    rwc_group,margin,_,_=vote(groups,rwc_weights(scores[quality]))
    return {"groups":groups,"representatives":representatives,"scores":scores,"selected":selected,
            "unique_leader":unique,"nonempty_leader":representatives[sc_group] is not None,
            "agreement":totals[sc_group]/len(rollouts),"quality_agrees":sc_group==rwc_group,"margin":margin}


def offline_question(raw,orders,dataset,qid):
    rows,states=[],[]
    for order_number,order in enumerate(orders):
        for n in BUDGETS:
            ids=order[:n]
            visible=[public_rollout(raw[i]) for i in ids]
            state=prefix_state(visible,dataset)
            total_cost=sum(r["retained_tokens"] for r in visible)
            oracle=any(raw[i]["correct"] for i in ids)
            choices=dict(state["selected"])
            if n==32:
                choices["shortest"]=int(np.argmin([r["retained_tokens"] for r in visible]))
                if dataset=="mathvista":
                    choices["token_fusion"]=select(state["groups"],state["scores"],"token_fusion")
                    choices["token_fusion_rwc"]=select(state["groups"],state["scores"],"token_fusion_rwc")
            for method,index in choices.items():
                candidate=raw[ids[index]]
                rows.append({"dataset":dataset,"sample_id":qid,"order":order_number,"N":n,"method":method,
                    "selected_rollout":int(ids[index]),"selected_answer":candidate["normalized_answer"],
                    "correct":bool(candidate["correct"]),"oracle":oracle,
                    "tokens":visible[0]["retained_tokens"] if method=="first" else total_cost,
                    "distinct_answers":len(state["representatives"]),"candidate_ids":ids,
                    "sampling_failure":not oracle,"selection_failure":bool(oracle and not candidate["correct"])})
            sc_index=state["selected"]["sc"]
            states.append({"dataset":dataset,"sample_id":qid,"order":order_number,"N":n,
                "unique_leader":state["unique_leader"],"nonempty_leader":state["nonempty_leader"],
                "agreement":state["agreement"],"quality_agrees":state["quality_agrees"],"margin":state["margin"],
                "correct":bool(raw[ids[sc_index]]["correct"]),"selected_answer":raw[ids[sc_index]]["normalized_answer"],
                "tokens":total_cost,"full_pool_correct_fraction":float(np.mean([r["correct"] for r in raw]))})
    return rows,states


def signal_diagnostics(raw,dataset,qid):
    from scipy.stats import spearmanr
    scores=mean_scores([public_rollout(r) for r in raw],vision=dataset=="mathvista")
    lengths=np.array([r["retained_tokens"] for r in raw])
    scores["length"]=lengths
    correct=np.array([r["correct"] for r in raw],dtype=float)
    rows=[]
    for name,values in scores.items():
        if name=="confidence_vision_correlation":
            valid=[x for x in values if x is not None and np.isfinite(x)]
            rows.append({"dataset":dataset,"sample_id":qid,"score":name,
                         "mean":float(np.mean(valid)) if valid else None,"valid_rollouts":len(valid)})
            continue
        order=np.argsort(-np.asarray(values),kind="stable")
        def corr(a,b):
            return float(spearmanr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else None
        rows.append({"dataset":dataset,"sample_id":qid,"score":name,
            "length_correlation":corr(lengths,values),"correctness_correlation":corr(values,correct),
            "top_quartile_accuracy":float(correct[order[:8]].mean()),
            "bottom_quartile_accuracy":float(correct[order[-8:]].mean()),
            "top_bottom_difference":float(correct[order[:8]].mean()-correct[order[-8:]].mean())})
    if dataset=="mathvista":
        rows.append({"dataset":dataset,"sample_id":qid,"score":"fusion_covariance",
                     "mean":float(np.mean(scores["token_fusion"]-scores["visconf"]))})
    return rows


def offline(root,run_id="main",commit=lambda:None):
    import pandas as pd
    root=Path(root)
    source=root/"data"/"task3"
    if not Store(root).done(source):
        raise RuntimeError("Run task3-prepare first")
    manifests=read_json(source/"manifest.json")
    sig=digest({"source":read_json(source/"DONE.json")["signature"],"code":code_hash("task3.py","scoring.py")})
    d=root/"runs"/run_id/"task3-offline"
    store=Store(root,commit)
    unit_store=Store(root)
    if store.done(d,sig):
        return
    all_rows,all_states,all_signals=[],[],[]
    for dataset,manifest in manifests.items():
        computed_new=0
        for i,qid in enumerate(manifest["question_ids"]):
            unit=d/dataset/digest(qid)[:20]
            usig=digest([sig,dataset,qid])
            if not unit_store.done(unit,usig):
                raw=read_parquet(source/dataset/digest(qid)[:20]/"rollouts.parquet")
                rows,states=offline_question(raw,manifest["orders"][qid],dataset,qid)
                signals=signal_diagnostics(raw,dataset,qid)
                split="development" if qid in manifest["splits"]["development"] else "evaluation"
                for collection in (rows,states,signals):
                    for row in collection:
                        row["split"]=split
                write_parquet(unit/"selections.parquet",rows)
                write_parquet(unit/"prefixes.parquet",states)
                write_parquet(unit/"signals.parquet",signals)
                unit_store.finish(unit,usig,[unit/"selections.parquet",unit/"prefixes.parquet",unit/"signals.parquet"])
                computed_new+=1
                if computed_new%50==0:
                    commit()
            all_rows.extend(read_parquet(unit/"selections.parquet"))
            all_states.extend(read_parquet(unit/"prefixes.parquet"))
            all_signals.extend(read_parquet(unit/"signals.parquet"))
            if (i+1)%25==0:
                print(f"{dataset}: offline questions {i+1}/1000",flush=True)
        if computed_new%50!=0:
            commit()
    write_parquet(d/"selections.parquet",all_rows)
    write_parquet(d/"prefixes.parquet",all_states)
    write_parquet(d/"signals.parquet",all_signals)
    df=pd.DataFrame(all_rows)
    # Average orders inside each question before any summary or bootstrap.
    perq=df.groupby(["dataset","split","sample_id","N","method"],sort=True)[
        ["correct","oracle","tokens","distinct_answers","sampling_failure","selection_failure"]].mean().reset_index()
    write_parquet(d/"question_means.parquet",perq.to_dict("records"))
    summary=perq.groupby(["dataset","split","N","method"],sort=True).agg(
        accuracy=("correct","mean"),oracle=("oracle","mean"),tokens=("tokens","mean"),
        distinct_answers=("distinct_answers","mean"),sampling_failure=("sampling_failure","mean"),
        selection_failure=("selection_failure","mean")).reset_index()
    write_parquet(d/"summary.parquet",summary.to_dict("records"))
    intervals=[]
    held=perq[perq.split=="evaluation"]
    for (dataset,n),part in held.groupby(["dataset","N"]):
        matrix=part.pivot(index="sample_id",columns="method",values="correct").sort_index()
        for method in matrix.columns:
            for baseline in ("first","sc"):
                if method!=baseline:
                    intervals.append({"dataset":dataset,"N":int(n),"method":method,"baseline":baseline,
                        **bootstrap_difference(matrix[method].to_numpy(),matrix[baseline].to_numpy())})
    write_json(d/"accuracy_differences.json",intervals)
    store.finish(d,sig,[d/f for f in ("selections.parquet","prefixes.parquet","signals.parquet",
        "question_means.parquet","summary.parquet","accuracy_differences.json")])


def policy_result(states,tau=None,gamma=None):
    # states contains only the revealed-prefix decision statistics. Correctness is
    # read after stopping for retrospective scoring, never to decide when to stop.
    for state in sorted(states,key=lambda r:r["N"]):
        n=state["N"]
        if n not in (4,8,16,32):
            continue
        if n==32:
            return state
        agreement=(tau is not None and state["unique_leader"] and state["nonempty_leader"] and state["agreement"]>=tau)
        if agreement and (gamma is None or (state["quality_agrees"] and state["margin"]>=gamma)):
            return state
    raise ValueError("Adaptive trace has no fixed-32 fallback")


def adaptive(root,run_id="main",commit=lambda:None):
    import pandas as pd
    root=Path(root)
    source=root/"runs"/run_id/"task3-offline"
    if not Store(root).done(source):
        raise RuntimeError("Run task3-offline first")
    d=root/"runs"/run_id/"task3-adaptive"
    sig=digest([read_json(source/"DONE.json")["signature"],code_hash("task3.py","scoring.py")])
    store=Store(root,commit)
    if store.done(d,sig):
        return
    traces=defaultdict(list)
    for row in read_parquet(source/"prefixes.parquet"):
        traces[(row["dataset"],row["split"],row["sample_id"],row["order"])].append(row)
    thresholds,grid_rows,held_rows=[],[],[]
    for dataset in ("math","mathvista"):
        development=[rows for (ds,split,_,_),rows in traces.items() if ds==dataset and split=="development"]
        baseline=float(np.mean([policy_result(rows)["correct"] for rows in development]))
        for family in ("agreement","quality"):
            candidates=[(None,None)]
            for tau in (.5,.625,.75,.875,1.0):
                for gamma in ((None,) if family=="agreement" else (0,.1,.25,.5)):
                    candidates.append((tau,gamma))
            scored=[]
            for tau,gamma in candidates:
                outcomes=[policy_result(rows,tau,gamma) for rows in development]
                accuracy=float(np.mean([r["correct"] for r in outcomes]))
                cost=float(np.mean([r["tokens"] for r in outcomes]))
                row={"dataset":dataset,"family":family,"tau":tau,"gamma":gamma,"accuracy":accuracy,
                    "tokens":cost,"fixed32_accuracy":baseline,"eligible":accuracy>=baseline-.01-1e-12}
                scored.append(row)
                grid_rows.append(row)
            feasible=[r for r in scored if r["eligible"]]
            chosen=min(feasible,key=lambda r:(r["tokens"],-r["accuracy"],
                       float("inf") if r["tau"] is None else r["tau"],
                       -1 if r["gamma"] is None else r["gamma"]))
            thresholds.append(chosen)
            for (ds,split,qid,order),rows in traces.items():
                if ds!=dataset or split!="evaluation":
                    continue
                result=policy_result(rows,chosen["tau"],chosen["gamma"])
                fixed=policy_result(rows)
                held_rows.append({"dataset":dataset,"family":family,"sample_id":qid,"order":order,
                    "N":result["N"],"tokens":result["tokens"],"correct":result["correct"],
                    "selected_answer":result["selected_answer"],"fixed32_correct":fixed["correct"],
                    "premature_error":result["N"]<32 and not result["correct"],
                    "recoverable_error":result["N"]<32 and not result["correct"] and fixed["correct"],
                    "full_pool_correct_fraction":result["full_pool_correct_fraction"]})
    write_json(d/"selected_policies.json",thresholds)
    write_parquet(d/"development_grid.parquet",grid_rows)
    write_parquet(d/"held_out.parquet",held_rows)
    df=pd.DataFrame(held_rows)
    summary=df.groupby(["dataset","family"]).agg(accuracy=("correct","mean"),tokens=("tokens","mean"),
        mean_rollouts=("N","mean"),premature_error=("premature_error","mean"),recoverable_error=("recoverable_error","mean")).reset_index()
    write_json(d/"summary.json",summary.to_dict("records"))
    distribution=df.groupby(["dataset","family","N"]).size().reset_index(name="count")
    write_json(d/"stopping_distribution.json",distribution.to_dict("records"))
    intervals=[]
    for (dataset,family),part in df.groupby(["dataset","family"]):
        q=part.groupby("sample_id")[["correct","fixed32_correct"]].mean()
        intervals.append({"dataset":dataset,"family":family,
            **bootstrap_difference(q.correct.to_numpy(),q.fixed32_correct.to_numpy())})
    write_json(d/"accuracy_differences.json",intervals)
    store.finish(d,sig,[d/f for f in ("selected_policies.json","development_grid.parquet","held_out.parquet",
        "summary.json","stopping_distribution.json","accuracy_differences.json")])


def live(root,run_id="main",smoke=False,commit=lambda:None):
    from .text import load_qwen,tokenizer_for,generate_text,cache_model
    root=Path(root)
    source=root/"data"/"task3"
    manifest=read_json(source/"manifest.json")["math"]
    d=root/"runs"/("live-smoke" if smoke else run_id)/"task3-live"
    sig=digest({"model":LIVE_MODEL,"revision":LIVE_REVISION,"seed":SEED,"smoke":smoke,
                "data":read_json(source/"DONE.json")["signature"],"code":code_hash("task3.py","scoring.py","text.py")})
    store=Store(root,commit)
    unit_store=Store(root)
    if store.done(d,sig):
        return
    model,tokenizer=None,None
    rows=[]
    question_ids=manifest["live_ids"][:1] if smoke else manifest["live_ids"]
    for qid in question_ids:
        raw=read_parquet(source/"math"/digest(qid)[:20]/"rollouts.parquet")[0]
        messages=[{"role":"system","content":"You are a helpful assistant."},
                  {"role":"user","content":raw["question_text"]+"\n\nThink step by step inside <think> </think> tags, then give your final answer inside \\boxed{}."}]
        responses={}
        generated_new=False
        for strategy,limit in (("parallel0",1024),("parallel1",1024),("parallel2",1024),("long",3072),("revision",2048)):
            unit=d/digest(qid)[:20]/strategy
            usig=digest([sig,qid,strategy])
            if not unit_store.done(unit,usig):
                if model is None:
                    model=load_qwen(root,name=LIVE_MODEL).eval()
                    tokenizer=tokenizer_for(root,LIVE_MODEL)
                prompt=messages
                if strategy=="revision":
                    prompt=messages+[{"role":"assistant","content":responses["parallel0"]["generated_text"]},
                        {"role":"user","content":"Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \\boxed{}."}]
                seed=int(digest([SEED,qid,strategy])[:8],16)
                result=generate_text(model,tokenizer,prompt,16 if smoke else limit,
                                     thinking=None,seed=seed,sampling=True)
                answer=last_boxed(result["generated_text"])
                result.update(sample_id=qid,strategy=strategy,extracted_answer=answer,
                              correct=math_equivalent(answer,str(raw["gold_answer"])))
                write_json(unit/"response.json",result)
                unit_store.finish(unit,usig,[unit/"response.json"])
                generated_new=True
            responses[strategy]=read_json(unit/"response.json")
        if generated_new:
            commit()
        alternatives=[responses[f"parallel{i}"] for i in range(3)]
        groups,_=group_answers([r["extracted_answer"] for r in alternatives],math_equivalent)
        selected=select(groups,{},"sc")
        for strategy,parts,winner in (("long",[responses["long"]],responses["long"]),
            ("parallel",alternatives,alternatives[selected]),
            ("refinement",[responses["parallel0"],responses["revision"]],responses["revision"])):
            rows.append({"sample_id":qid,"strategy":strategy,"correct":winner["correct"],
                "input_tokens":sum(r["input_tokens"] for r in parts),"output_tokens":sum(r["output_tokens"] for r in parts),
                "seconds":sum(r["seconds"] for r in parts),"extracted_answer":winner["extracted_answer"],
                "extraction_failure":winner["extracted_answer"] is None,
                "truncated_calls":sum(r["hit_max_new_tokens"] for r in parts),"calls":len(parts),
                "peak_gpu_bytes":max(r["peak_gpu_bytes"] for r in parts),
                "initial_correct":responses["parallel0"]["correct"],
                "wrong_to_correct":strategy=="refinement" and not responses["parallel0"]["correct"] and winner["correct"],
                "correct_to_wrong":strategy=="refinement" and responses["parallel0"]["correct"] and not winner["correct"]})
    write_parquet(d/"strategies.parquet",rows)
    import pandas as pd
    summary=pd.DataFrame(rows).groupby("strategy").agg(accuracy=("correct","mean"),input_tokens=("input_tokens","sum"),
        output_tokens=("output_tokens","sum"),seconds=("seconds","sum"),extraction_failure=("extraction_failure","mean"),
        truncated_calls=("truncated_calls","sum"),calls=("calls","sum"),peak_gpu_bytes=("peak_gpu_bytes","max"),
        wrong_to_correct=("wrong_to_correct","sum"),correct_to_wrong=("correct_to_wrong","sum")).reset_index()
    write_json(d/"summary.json",summary.to_dict("records"))
    store.finish(d,sig,[d/"strategies.parquet",d/"summary.json"])

