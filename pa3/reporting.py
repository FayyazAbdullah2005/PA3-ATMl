"""Readable data exports, figures and review galleries; no interpretation/report text."""
from __future__ import annotations
import csv
import html
import json
from pathlib import Path
import shutil
import numpy as np
from .io import *


def csv_export(path,rows):
    rows=clean_json(list(rows))
    if not rows:
        return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with atomic_path(path) as tmp, tmp.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            cells={k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()}
            # Model text is untrusted spreadsheet content. Keep original strings
            # in JSON/Parquet; prevent Excel from executing CSV cells as formulas.
            writer.writerow({k:("'"+v if isinstance(v,str) and v.lstrip().startswith(("=","+","-","@")) else v) for k,v in cells.items()})


def page(path,title,body):
    css="body{max-width:1400px;margin:30px auto;padding:0 20px;font:16px/1.5 system-ui;color:#20242a}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:10px;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f7;padding:14px}img{max-width:440px;max-height:350px}h2{margin-top:40px}a{color:#1459a5}.warning{background:#fff3cd;padding:12px}"
    with atomic_path(path) as tmp:
        tmp.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{css}</style></head><body><h1>{html.escape(title)}</h1>{body}</body></html>',encoding="utf-8")


def plot_lines(path,series,xlabel,ylabel,boundaries=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.8),layout="constrained")
    for label,x,y in series:
        ax.plot(x,y,label=label,marker="o" if len(x)<10 else None,linewidth=1.6)
    ax.set(xlabel=xlabel,ylabel=ylabel)
    ax.grid(alpha=.22)
    for i,x in enumerate(boundaries,1):
        ax.axvline(x,color="#666666",linestyle="--",alpha=.6,linewidth=.8)
        ax.annotate(f"Stage {i} end",xy=(x,1),xycoords=("data","axes fraction"),
                    rotation=90,fontsize=7,ha="right",va="top")
    if series:
        ax.legend(fontsize=8)
    path.parent.mkdir(parents=True,exist_ok=True)
    for suffix in (".png",".pdf"):
        with atomic_path(path.with_suffix(suffix)) as tmp:
            fig.savefig(tmp,format=suffix[1:],dpi=180)
    plt.close(fig)


def label_gallery(root,output):
    root,output=Path(root),Path(output)
    d=root/"data"/"task2"
    if not (d/"pairs.json").exists():
        return
    pairs=read_json(d/"pairs.json")
    index=read_json(d/"image_index.json")
    body=[]
    reviews=[]
    for suite in ("grounding_dev","grounding_test","natural_dev","natural_test"):
        body.append(f"<h2>{suite}</h2>")
        for p in pairs[suite]:
            cells=[]
            for e in p["examples"]:
                image=root/"data"/"task2"/"images"/index[str(e["image_id"])]["file"]
                target=output/"images"/image.name
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    shutil.copy2(image,target)
                cells.append(f'<td><img src="images/{html.escape(image.name)}"><p>Image {e["image_id"]}</p><p>Reference: {html.escape(e["answer"])}</p></td>')
            body.append(f'<h3>{html.escape(p["pair_id"])}</h3><p>{html.escape(p["examples"][0]["question"])}</p><table><tr>{"".join(cells)}</tr></table>')
            reviews.append({"suite":suite,"pair_id":p["pair_id"],"label_valid":"","ambiguity":"","notes":""})
    page(output/"label_review.html","Task 2 label review", "".join(body))
    if not (output/"label_review.csv").exists():
        csv_export(output/"label_review.csv",reviews)


def summarize(root,run_id="main",commit=lambda:None):
    import pandas as pd
    from .config import T1_METHODS, T1_TASKS
    root=Path(root)
    source=root/"runs"/run_id
    output=root/"exports"/run_id
    output.mkdir(parents=True,exist_ok=True)
    links=[]
    def link(path,label=None):
        rel=path.relative_to(output).as_posix()
        links.append(f'<li><a href="{html.escape(rel)}">{html.escape(label or rel)}</a></li>')
    # Export every compact numerical table as UTF-8 CSV, retaining Parquet too.
    for path in source.rglob("*.parquet") if source.exists() else []:
        # Per-question/per-generation shards are represented by their aggregate table.
        rel=path.relative_to(source)
        if any(part.startswith("batch-") for part in rel.parts) or any(len(part)==20 and all(c in "0123456789abcdef" for c in part) for part in rel.parts):
            continue
        rows=read_parquet(path)
        target=output/rel.with_suffix(".csv")
        csv_export(target,rows)
        shutil.copy2(path,target.with_suffix(".parquet"))
        link(target)
    for path in source.rglob("*.json") if source.exists() else []:
        if path.name not in {"summary.json","accuracy_differences.json","selected_policies.json",
                             "correct_image_advantages.json","stopping_distribution.json","resources.json",
                             "displacement.json","merge_check.json","calibration.json","run.json","completed.json"}:
            continue
        rel=path.relative_to(source)
        if any(part.startswith("step-") for part in rel.parts):
            continue
        target=output/rel
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        value=read_json(path)
        if isinstance(value,list) and value and isinstance(value[0],dict):
            csv_export(target.with_suffix(".csv"),value)
        link(target)
    # Export data-selection, split/example ID, and scorer manifests for the repository.
    for rel in ("pins.json","task1/selection.json","task1/probe_ids.json",
                "task2/split_manifest.json","task3/manifest.json","task3/scorer.json"):
        src=root/"data"/rel
        if src.exists():
            dst=output/"manifests"/rel
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst)
            link(dst)
    # Task 1 benchmark table, GSM8K extraction/truncation diagnostics, and baseline percentage-point differences.
    benchmark_rows=[]
    gsm8k_example_rows=[]
    conditions=["baseline","full-mid",*T1_METHODS]
    primary={"gsm8k_cot":"exact_match,flexible-extract","hellaswag":"acc_norm,none",
             "boolq":"acc,none","arc_easy":"acc_norm,none"}
    for condition in conditions:
        row={"condition":condition}
        for task in T1_TASKS:
            path=source/"task1-evaluation"/condition/task/"scores.json"
            if path.exists():
                scores=read_json(path)["results"][task]
                row[task]=100*scores[primary[task]]
                row[task+"_raw_accuracy"]=100*scores["acc,none"] if "acc,none" in scores else None
                if task=="gsm8k_cot" and "exact_match,strict-match" in scores:
                    row["gsm8k_cot_strict_match"]=100*scores["exact_match,strict-match"]
            else:
                row[task]=None
        examples_path=source/"task1-evaluation"/condition/"gsm8k_cot"/"examples.json"
        if examples_path.exists():
            samples=read_json(examples_path)
            truncated=sum(bool(s.get("hit_max_new_tokens")) for s in samples)
            unextractable=sum(s.get("extracted_answer") in (None,"","[invalid]") for s in samples)
            row["gsm8k_truncated_count"]=truncated
            row["gsm8k_truncation_rate"]=100*truncated/len(samples) if samples else None
            row["gsm8k_unextractable_count"]=unextractable
            row["gsm8k_unextractable_rate"]=100*unextractable/len(samples) if samples else None
            for s in samples:
                gsm8k_example_rows.append({"condition":condition,"doc_id":s.get("doc_id"),
                    "target":s.get("target"),"extracted_answer":s.get("extracted_answer"),
                    "exact_match":s.get("exact_match"),"hit_max_new_tokens":s.get("hit_max_new_tokens"),
                    "input_tokens":s.get("input_tokens"),"output_tokens":s.get("output_tokens")})
        benchmark_rows.append(row)
    for row in benchmark_rows:
        for task in T1_TASKS:
            base=benchmark_rows[0][task]
            row[task+"_delta_pp"]=row[task]-base if row[task] is not None and base is not None else None
    csv_export(output/"task1_benchmarks.csv",benchmark_rows)
    link(output/"task1_benchmarks.csv")
    if gsm8k_example_rows:
        csv_export(output/"task1_gsm8k_examples.csv",gsm8k_example_rows)
        link(output/"task1_gsm8k_examples.csv")
    curves=[]
    resources=[]
    for method in T1_METHODS:
        d=source/"task1"/method
        records=read_events(d/"training.jsonl")
        # A resumed run can have duplicate logs beyond a checkpoint; latest wins.
        records=sorted({r["step"]:r for r in records}.values(),key=lambda r:r["step"])
        if records:
            curves.append((method,[r["step"] for r in records],[r["sft_loss"] for r in records]))
            csv_export(output/"task1"/f"{method}_training.csv",records)
            link(output/"task1"/f"{method}_training.csv")
        if (d/"resources.json").exists():
            resource={"method":method,**read_json(d/"resources.json")}
            if records:
                resource.update(training_seconds=records[-1]["training_seconds"],peak_gpu_bytes=max(r["peak_gpu_bytes"] for r in records))
            milestones=sorted((d/"checkpoints").glob("step-*.pt"))
            resource["checkpoint_bytes"]=sum(p.stat().st_size for p in milestones)
            resource["single_checkpoint_bytes"]=milestones[-1].stat().st_size if milestones else 0
            disp=source/"task1-diagnostics"/method/"displacement.json"
            if disp.exists():
                resource["l2_displacement"]=read_json(disp).get("l2_displacement")
            if method=="full" and (source/"task1-diagnostics"/"full-mid"/"displacement.json").exists():
                resource["l2_displacement_midpoint"]=read_json(source/"task1-diagnostics"/"full-mid"/"displacement.json").get("l2_displacement")
            if (d/"merge_check.json").exists():
                resource.update(read_json(d/"merge_check.json"))
            resources.append(resource)
    if curves:
        plot_lines(output/"figures"/"task1_sft",curves,"Optimizer update","Response-token cross-entropy")
        link(output/"figures"/"task1_sft.png")
    csv_export(output/"task1_resources.csv",resources)
    if resources:
        link(output/"task1_resources.csv")
    diagnostic_rows=[]
    for path in (source/"task1-diagnostics").glob("*/diagnostics.parquet"):
        diagnostic_rows.extend(read_parquet(path))
    if diagnostic_rows:
        df=pd.DataFrame(diagnostic_rows)
        compact=df.groupby(["condition","layer","group"],dropna=False).agg(
            rho=("rho","first"),eta=("eta","first"),R_mean=("R","mean"),R_median=("R","median"),
            R_q25=("R",lambda x:x.quantile(.25)),R_q75=("R",lambda x:x.quantile(.75))).reset_index()
        csv_export(output/"task1_diagnostics_summary.csv",compact.to_dict("records"))
        link(output/"task1_diagnostics_summary.csv")
    penalty=read_events(source/"task1"/"l2sp"/"training.jsonl")
    penalty=sorted({r["step"]:r for r in penalty}.values(),key=lambda r:r["step"])
    if penalty:
        plot_lines(output/"figures"/"task1_l2sp_objective",[(name,[r["step"] for r in penalty],[r[name] for r in penalty])
            for name in ("sft_loss","weighted_penalty","total_loss")],"Optimizer update","Loss / weighted penalty")
        link(output/"figures"/"task1_l2sp_objective.png")
    # Aligned qualitative responses and the exact manual annotation subset.
    responses=[]
    for condition in conditions:
        path=source/"task1-qualitative"/condition/"responses.json"
        if path.exists():
            responses.extend(read_json(path))
    if responses:
        body=[]
        for prompt in dict.fromkeys(r["prompt_id"] for r in responses):
            body.append(f"<h2>{html.escape(prompt)}</h2>")
            original=next(r for r in responses if r["prompt_id"]==prompt)
            body.append(f'<pre>{html.escape(original["messages"][-1]["content"])}</pre>')
            for row in [r for r in responses if r["prompt_id"]==prompt]:
                body.append(f'<h3>{html.escape(row["condition"])}</h3><pre>{html.escape(row["generated_text"])}</pre><p>Truncated: {row["hit_max_new_tokens"]}</p>')
        page(output/"task1_responses.html","Task 1 aligned responses","".join(body))
        csv_export(output/"task1_responses.csv",responses)
        annotation=[]
        for r in responses:
            if r["prompt_id"].startswith(("what_is_your_wish","enough_of_my_husband")):
                annotation.append({"prompt_id":r["prompt_id"],"condition":r["condition"],"response":r["generated_text"],
                    "truncated":r["hit_max_new_tokens"],"format_compliance":"","coherence_relevance":"",
                    "repetition":"","mathematical_framing":"","inappropriate_content":"","excerpt":"","notes":""})
        # Preserve existing manual fields, including during a partial-results export.
        annotation_path=output/"task1_annotations.csv"
        if annotation_path.exists():
            with annotation_path.open(encoding="utf-8-sig",newline="") as f:
                old={(r["prompt_id"],r["condition"]):r for r in csv.DictReader(f)}
            for row in annotation:
                prior=old.get((row["prompt_id"],row["condition"]),{})
                for field in ("format_compliance","coherence_relevance","repetition","mathematical_framing","inappropriate_content","excerpt","notes"):
                    row[field]=prior.get(field,"")
        csv_export(annotation_path,annotation)
        link(output/"task1_responses.html")
        link(output/"task1_annotations.csv")
    t2_resources=[]
    feature_shards=sorted((root/"data"/"task2"/"features").glob("shard-*/resources.json"))
    if feature_shards:
        shard_rows=[read_json(p) for p in feature_shards]
        caching={"images":sum(r["images"] for r in shard_rows),
                 "seconds":sum(r["seconds"] for r in shard_rows),
                 "peak_gpu_bytes":max(r["peak_gpu_bytes"] for r in shard_rows),
                 "feature_bytes":sum(p.stat().st_size for p in (root/"data"/"task2"/"features").glob("shard-*/features.npy"))}
        write_json(output/"task2_feature_caching.json",caching)
        link(output/"task2_feature_caching.json")
        t2_resources.append({"tokens":196,"stage":"feature_caching","update":0,
            "training_seconds":caching["seconds"],"peak_gpu_bytes":caching["peak_gpu_bytes"],
            "trainable_parameters":0,"images":caching["images"],"feature_bytes":caching["feature_bytes"]})
    for tokens in (196,49):
        directory=source/"task2"/str(tokens)
        logs=read_events(directory/"training.jsonl")
        logs=sorted({r["step"]:r for r in logs}.values(),key=lambda r:r["step"])
        if logs:
            csv_export(output/"task2"/str(tokens)/"training.csv",logs)
            link(output/"task2"/str(tokens)/"training.csv")
            plot_lines(output/"figures"/f"task2_{tokens}_loss",[(str(tokens),[r["step"] for r in logs],[r["sft_loss"] for r in logs])],"Optimizer update","Example-weighted cross-entropy",(2250,3750))
            link(output/"figures"/f"task2_{tokens}_loss.png")
        for step_name in ("2250","3750","4500"):
            ep=directory/"endpoints"/step_name/"resources.json"
            if ep.exists():
                info=read_json(ep)
                cal=read_json(directory/"calibration.json") if (directory/"calibration.json").exists() else {}
                t2_resources.append({"tokens":tokens,"stage":f"endpoint_{step_name}",**info,
                                     "visual_scale":cal.get("scale")})
        development=[]
        for path in sorted((directory/"development").glob("step-*/summary.json")):
            row=read_json(path)
            development.append({"update":row["update"],**row["metrics"]})
        if development:
            csv_export(output/"task2"/str(tokens)/"development.csv",development)
            link(output/"task2"/str(tokens)/"development.csv")
            series=[(metric,[r["update"] for r in development],[r[metric] for r in development]) for metric in development[0] if metric!="update"]
            plot_lines(output/"figures"/f"task2_{tokens}_development",series,"Optimizer update","Reference loss / NLL",(2250,3750))
            link(output/"figures"/f"task2_{tokens}_development.png")
        evaluation=source/"task2-evaluation"/str(tokens)/"examples.parquet"
        if evaluation.exists():
            records=read_parquet(evaluation)
            index=read_json(root/"data"/"task2"/"image_index.json")
            body=[]
            # Deterministic rule: first three sorted pair IDs per suite, identical
            # across both visual-token variants and independent of model results.
            for suite in sorted({r["suite"] for r in records}):
                pair_ids=sorted({r["pair_id"] for r in records if r["suite"]==suite})
                for pid in pair_ids[:3]:
                    chosen=[r for r in records if r["suite"]==suite and r["pair_id"]==pid]
                    body.append(f"<h2>{html.escape(suite+' / '+pid)}</h2><p>{html.escape(chosen[0]['question'])}</p>")
                    for image_id in sorted({r["image_id"] for r in chosen}):
                        info=index[str(image_id)]
                        image=root/"data"/"task2"/"images"/info["file"]
                        target=output/"images"/image.name
                        target.parent.mkdir(parents=True,exist_ok=True)
                        if not target.exists():
                            shutil.copy2(image,target)
                        body.append(f'<img src="images/{image.name}"><table><tr><th>Condition</th><th>Original reference</th><th>Prediction</th><th>Score</th></tr>')
                        for r in chosen:
                            if r["image_id"]==image_id:
                                body.append(f'<tr><td>{r["condition"]}</td><td>{html.escape(r["answer"])}</td><td>{html.escape(r["prediction"])}</td><td>{r["score"]:.3f}</td></tr>')
                        body.append("</table>")
            page(output/f"task2_{tokens}_examples.html",f"Task 2: {tokens} visual tokens","".join(body))
            link(output/f"task2_{tokens}_examples.html")
    if t2_resources:
        csv_export(output/"task2_resources.csv",t2_resources)
        link(output/"task2_resources.csv")
    t3=source/"task3-offline"/"summary.parquet"
    if t3.exists():
        df=pd.DataFrame(read_parquet(t3))
        for dataset,part in df[df.split=="evaluation"].groupby("dataset"):
            for x,y in (("N","accuracy"),("tokens","accuracy"),("N","oracle")):
                series=[]
                for method,group in part.groupby("method"):
                    if y=="oracle" and method!="sc":
                        continue
                    group=group.sort_values("N")
                    series.append(("Oracle coverage" if y=="oracle" else method,group[x].tolist(),group[y].tolist()))
                name=output/"figures"/f"task3_{dataset}_{y}_vs_{x}"
                plot_lines(name,series,"Candidates" if x=="N" else "Retained generated tokens",y.replace("_"," ").title())
                link(name.with_suffix(".png"))
        signal_path=source/"task3-offline"/"signals.parquet"
        if signal_path.exists():
            signals=pd.DataFrame(read_parquet(signal_path))
            numeric=[c for c in signals if c not in ("dataset","split","sample_id","score")]
            compact=signals.groupby(["dataset","split","score"])[numeric].mean().reset_index()
            csv_export(output/"task3_signal_summary.csv",compact.to_dict("records"))
            link(output/"task3_signal_summary.csv")
        # Fixed illustrative trajectories: first three held-out IDs and first
        # two rollout IDs of their first saved order, chosen without scores.
        manifests=read_json(root/"data"/"task3"/"manifest.json")
        selections=read_parquet(source/"task3-offline"/"selections.parquet")
        examples_body=[]
        for dataset,manifest in manifests.items():
            for qid in manifest["splits"]["evaluation"][:3]:
                raw=read_parquet(root/"data"/"task3"/dataset/digest(qid)[:20]/"rollouts.parquet")
                if dataset=="mathvista":
                    low,high=np.quantile(np.concatenate([r["kl_u_p"] for r in raw]),[.05,.95])
                for rollout in manifest["orders"][qid][0][:2]:
                    row=raw[rollout]
                    trajectory=[{"token":i,"logp":p,"kl_u_p":k} for i,(p,k) in enumerate(zip(row["logp"],row["kl_u_p"]))]
                    names=["logp","kl_u_p"]
                    if dataset=="mathvista":
                        names=["calibrated_confidence","visual_engagement"]
                        for i,value in enumerate(trajectory):
                            denom=row["img_attn_total"][i]+row["prompt_text_attn_total"][i]+row["generated_text_attn_total"][i]
                            value["calibrated_confidence"]=float(np.clip((value["kl_u_p"]-low)/max(high-low,1e-8),0,1))
                            value["visual_engagement"]=row["img_attn_total"][i]/denom
                    name=f"{dataset}_{digest(qid)[:10]}_rollout{rollout}"
                    csv_export(output/"trajectories"/(name+".csv"),trajectory)
                    # Rolling means make long token trajectories legible; raw values
                    # remain in the CSV, with a stated fixed window of 32 tokens.
                    frame=pd.DataFrame(trajectory)
                    plot_lines(output/"figures"/name,[(signal,frame.token.tolist(),frame[signal].rolling(32,min_periods=1).mean().tolist()) for signal in names],
                               "Generated token index","Signal (32-token rolling mean)")
                    link(output/"figures"/(name+".png"))
            # One deterministic selection disagreement per dataset, if available.
            rows=[r for r in selections if r["dataset"]==dataset and r["split"]=="evaluation" and r["N"]==32 and r["order"]==0]
            by_question={}
            for row in rows:
                by_question.setdefault(row["sample_id"],[]).append(row)
            chosen=next((qid for qid in sorted(by_question) if len({r["selected_answer"] for r in by_question[qid]})>1),None)
            if chosen:
                raw=read_parquet(root/"data"/"task3"/dataset/digest(chosen)[:20]/"rollouts.parquet")
                examples_body.append(f'<h2>{html.escape(dataset+" / "+chosen)}</h2><pre>{html.escape(raw[0]["question_text"])}</pre>')
                for selection in by_question[chosen]:
                    response=raw[selection["selected_rollout"]]
                    examples_body.append(f'<h3>{selection["method"]} / rollout {selection["selected_rollout"]}</h3><p>Selected: {html.escape(str(selection["selected_answer"]))}; correct: {selection["correct"]}</p><pre>{html.escape(response["generated_text"])}</pre>')
        page(output/"task3_selection_examples.html","Task 3 selection disagreement",'<p>Rule: first sorted held-out question with selection disagreement at N=32, order 0.</p>'+"".join(examples_body))
        link(output/"task3_selection_examples.html")
    # Explicitly selected revision and premature-stopping cases for manual review.
    case_rows=[]
    cases_body=[]
    live_path=source/"task3-live"/"strategies.parquet"
    if live_path.exists():
        live=read_parquet(live_path)
        for label in ("wrong_to_correct","correct_to_wrong"):
            matches=sorted((r for r in live if r["strategy"]=="refinement" and r[label]),key=lambda r:r["sample_id"])
            record=matches[0] if matches else None
            details=None
            if record:
                qid=record["sample_id"]
                qdir=source/"task3-live"/digest(qid)[:20]
                p0=read_json(qdir/"parallel0"/"response.json") if (qdir/"parallel0"/"response.json").exists() else {}
                rev=read_json(qdir/"revision"/"response.json") if (qdir/"revision"/"response.json").exists() else {}
                rollout_path=root/"data"/"task3"/"math"/digest(qid)[:20]/"rollouts.parquet"
                gold=read_parquet(rollout_path)[0]["gold_answer"] if rollout_path.exists() else None
                details={"sample_id":qid,"gold_answer":gold,
                         "initial_answer":p0.get("extracted_answer"),"initial_text":p0.get("generated_text"),
                         "revised_answer":rev.get("extracted_answer"),"revised_text":rev.get("generated_text")}
                cases_body.append(f"<h2>Live refinement: {html.escape(label)} ({html.escape(qid)})</h2>"
                    f"<p>Gold: {html.escape(str(gold))} | Initial: {html.escape(str(details['initial_answer']))} | Revised: {html.escape(str(details['revised_answer']))}</p>"
                    f"<h3>Initial response (parallel0)</h3><pre>{html.escape(str(details['initial_text']))}</pre>"
                    f"<h3>Critique and revision</h3><pre>{html.escape(str(details['revised_text']))}</pre>")
            case_rows.append({"case":label,"available":bool(matches),"record":record,"details":details})
    early_path=source/"task3-adaptive"/"held_out.parquet"
    if early_path.exists():
        early=read_parquet(early_path)
        prefixes=read_parquet(source/"task3-offline"/"prefixes.parquet") if (source/"task3-offline"/"prefixes.parquet").exists() else []
        for dataset in ("math","mathvista"):
            matches=sorted((r for r in early if r["dataset"]==dataset and r["recoverable_error"]),
                           key=lambda r:(r["sample_id"],r["order"],r["family"]))
            record=matches[0] if matches else None
            details=None
            if record:
                qid=record["sample_id"]
                rollout_path=root/"data"/"task3"/dataset/digest(qid)[:20]/"rollouts.parquet"
                raw=read_parquet(rollout_path) if rollout_path.exists() else []
                trace=[p for p in prefixes if p["dataset"]==dataset and p["sample_id"]==qid and p["order"]==record["order"]]
                details={"question_text":raw[0]["question_text"] if raw else None,
                         "gold_answer":raw[0]["gold_answer"] if raw else None,"prefix_trace":trace}
                cases_body.append(f"<h2>Recoverable early stop: {html.escape(dataset)} / {html.escape(qid)} (order {record['order']}, {html.escape(record['family'])})</h2>"
                    f"<p>Stopped at N={record['N']} with answer {html.escape(str(record['selected_answer']))}; Gold: {html.escape(str(details['gold_answer']))}</p>"
                    f"<pre>{html.escape(str(details['question_text']))}</pre>")
            case_rows.append({"case":dataset+"_recoverable_early_stop","available":bool(matches),"record":record,"details":details})
    if case_rows:
        write_json(output/"task3_manual_case_index.json",case_rows)
        link(output/"task3_manual_case_index.json")
        if cases_body:
            page(output/"task3_cases.html","Task 3 refinement and stopping cases","".join(cases_body))
            link(output/"task3_cases.html")
    # A complete file inventory makes omissions explicit without inventing results.
    expected={"task1_benchmarks":28,"task1_qualitative_responses":168,"task2_final_responses":2560,
              "task3_live_unique_responses":250}
    actual={"task1_benchmarks":sum((source/"task1-evaluation"/c/t/"DONE.json").exists() for c in conditions for t in T1_TASKS),
            "task1_qualitative_responses":len(responses),
            "task2_final_responses":sum(len(read_parquet(p)) for p in (source/"task2-evaluation").glob("*/examples.parquet")),
            "task3_live_unique_responses":len(list((source/"task3-live").glob("*/*/response.json")))}
    inventory={"expected":expected,"actual":actual,"complete_counts":expected==actual,
               "completed_units":Store(root).status(),"run_id":run_id,
               "manual_work":"Qualitative annotations, interpretation, report text and LaTeX are student work."}
    write_json(output/"inventory.json",inventory)
    counts='<table><tr><th>Evidence</th><th>Expected</th><th>Available</th></tr>'+"".join(f'<tr><td>{k}</td><td>{expected[k]}</td><td>{actual[k]}</td></tr>' for k in expected)+"</table>"
    page(output/"index.html","ATML PA3 results",'<p>Machine-readable tables, figures, and raw responses. Missing experiments remain blank; no scores are fabricated.</p>'+counts+'<ul>'+"".join(links)+"</ul>")
    import zipfile
    bundle=root/"exports"/(run_id+".zip")
    with atomic_path(bundle) as tmp,zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.rglob("*")):
            if file.is_file():
                archive.write(file,file.relative_to(output).as_posix())
    commit()
    print(f"Readable results: {output/'index.html'}",flush=True)

