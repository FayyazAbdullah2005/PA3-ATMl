"""COCO/VQAv2 selection, reusable grounding pools, and split-isolation checks."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import numpy as np
from .config import SEED, QWEN, SIGLIP
from .io import *
from .scoring import extract_contractions, vqa_normalize
from .text import ensure_pins, cache_model, tokenizer_for

VQA_BASE = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/"
COCO_BASE = "https://s3.amazonaws.com/images.cocodataset.org/"
PLURALS = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep",
           "skis": "skis", "scissors": "scissors", "toothbrush": "toothbrushes", "bench": "benches",
           "bus": "buses", "sandwich": "sandwiches", "wine glass": "wine glasses"}


def plural(name):
    if name in PLURALS:
        return PLURALS[name]
    return name+"es" if name.endswith(("s", "x", "ch", "sh")) else name+"s"


def read_zip_json(path, contains=None):
    import zipfile
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".json") and (contains is None or contains in n)]
        if len(names) != 1:
            raise ValueError(f"Expected one matching JSON in {path}, got {names}")
        with z.open(names[0]) as f:
            return json.load(f)


def natural_candidates(questions, annotations, pair_list, tokenizer, contractions):
    qs = {q["question_id"]: q for q in questions["questions"]}
    anns = {a["question_id"]: a for a in annotations["annotations"]}
    if isinstance(pair_list, dict):
        for key in ("complementary_pairs", "pairs"):
            if key in pair_list:
                pair_list = pair_list[key]
                break
    candidates, seen = [], set()
    for pair in pair_list:
        if isinstance(pair, dict):
            pair = pair.get("question_ids") or [pair["question_id"], pair["complementary_question_id"]]
        ids = tuple(sorted(map(int, pair)))
        if len(ids) != 2 or ids in seen:
            continue
        seen.add(ids)
        records = []
        for qid in ids:
            q, a = qs[qid], anns[qid]
            refs = [x["answer"] for x in a["answers"]]
            normalized = [vqa_normalize(x, contractions) for x in refs]
            consensus, count = Counter(normalized).most_common(1)[0]
            if count < 8 or not consensus or len(refs) != 10:
                break
            answer = next(x for x, n in zip(refs, normalized) if n == consensus)
            if len(tokenizer.encode(q["question"], add_special_tokens=False)) > 80 or len(tokenizer.encode(answer, add_special_tokens=False)) > 16:
                break
            records.append({"question_id": qid, "image_id": q["image_id"], "question": q["question"],
                "answer": answer, "canonical": consensus, "references": refs, "type": a["answer_type"]})
        if len(records) != 2 or records[0]["image_id"] == records[1]["image_id"]:
            continue
        if records[0]["question"] != records[1]["question"] or records[0]["canonical"] == records[1]["canonical"]:
            continue
        if records[0]["type"] != records[1]["type"]:
            continue
        candidates.append({"pair_id": "natural-"+"-".join(map(str, ids)), "type": records[0]["type"], "examples": records})
    return sorted(candidates, key=lambda x: x["pair_id"])


def select_natural(candidates, quotas, rng, excluded=None, unique_images=False):
    selected, used = [], set(excluded or ())
    for kind, count in quotas.items():
        pool = [p for p in candidates if p["type"] == kind]
        found = 0
        for index in rng.permutation(len(pool)):
            p = pool[int(index)]
            images = {e["image_id"] for e in p["examples"]}
            if images & used:
                continue
            selected.append(p)
            if unique_images:
                used.update(images)
            found += 1
            if found == count:
                break
        if found != count:
            raise ValueError(f"Insufficient eligible {kind} complementary pairs: {found}/{count}")
    return selected


def object_pools(instances, image_ids):
    wanted = set(image_ids)
    counts = defaultdict(Counter)
    crowd = defaultdict(set)
    for ann in instances["annotations"]:
        if ann["image_id"] in wanted:
            counts[ann["category_id"]][ann["image_id"]] += 1
            if ann.get("iscrowd", 0):
                crowd[ann["category_id"]].add(ann["image_id"])
    pools = {}
    for category in instances["categories"]:
        cid = category["id"]
        groups = {str(n): sorted(i for i, c in counts[cid].items() if c == n and i not in crowd[cid]) for n in range(1, 5)}
        positive = sorted(counts[cid])
        negative = sorted(wanted-set(positive))
        if positive and negative and sum(bool(v) for v in groups.values()) >= 2:
            pools[str(cid)] = {"name": category["name"], "positive": positive, "negative": negative, "counts": groups}
    if not pools:
        raise ValueError("No valid grounding category pools")
    return pools


def draw_grounding(pools, kind, rng, pair_id="sample", excluded=None):
    excluded = set(excluded or ())
    eligible = []
    for cid, p in sorted(pools.items()):
        if kind == "presence":
            groups = [[i for i in p[k] if i not in excluded] for k in ("positive", "negative")]
            if all(groups):
                eligible.append((cid, p, groups))
        else:
            groups = {n: [i for i in ids if i not in excluded] for n, ids in p["counts"].items()}
            groups = {n: ids for n, ids in groups.items() if ids}
            if len(groups) >= 2:
                eligible.append((cid, p, groups))
    if not eligible:
        raise ValueError("Grounding pool exhausted by image-isolation constraints")
    cid, pool, groups = eligible[int(rng.integers(len(eligible)))]
    if kind == "presence":
        image_ids = [int(rng.choice(g)) for g in groups]
        answers = ["yes", "no"]
        question = f"Does this image contain {pool['name']}?"
    else:
        numbers = rng.choice(sorted(groups), 2, replace=False)
        image_ids = [int(rng.choice(groups[n])) for n in numbers]
        answers = list(map(str, numbers))
        question = f"How many {plural(pool['name'])} can you see?"
    examples = [{"image_id": i, "question": question, "answer": a, "canonical": a,
                 "references": [a], "type": kind, "question_id": f"{pair_id}-{j}"}
                for j, (i, a) in enumerate(zip(image_ids, answers))]
    return {"pair_id": pair_id, "type": kind, "category_id": cid, "examples": examples}


def pair_images(pairs):
    return {e["image_id"] for p in pairs for e in p["examples"]}


def prepare(root, commit=lambda: None):
    root = Path(root)
    store = Store(root, commit)
    pins = ensure_pins(root, commit)
    cache_model(root, QWEN, commit)
    cache_model(root, SIGLIP, commit)
    d, downloads = root/"data"/"task2", root/"data"/"downloads"
    signature = digest({"pins": pins, "code": code_hash("data2.py", "scoring.py", "text.py")})
    if store.done(d, signature):
        return
    d.mkdir(parents=True, exist_ok=True)
    tok = tokenizer_for(root)
    official = download("https://raw.githubusercontent.com/GT-Vision-Lab/VQA/master/PythonEvaluationTools/vqaEvaluation/vqaEval.py", downloads/"vqaEval.py")
    contractions = extract_contractions(official.read_text(encoding="utf-8"))
    write_json(d/"contractions.json", contractions)
    archive = download(COCO_BASE+"annotations/annotations_trainval2014.zip", downloads/"annotations_trainval2014.zip")
    captions = {split: read_zip_json(archive, f"captions_{split}2014") for split in ("train", "val")}
    instances = {split: read_zip_json(archive, f"instances_{split}2014") for split in ("train", "val")}
    rng = np.random.default_rng(SEED)
    natural = {}
    for split, title in (("train", "Train"), ("val", "Val")):
        values = {}
        for kind in ("Questions", "Annotations", "Complementary_Pairs"):
            name = f"v2_{kind}_{title}_mscoco.zip"
            values[kind] = read_zip_json(download(VQA_BASE+name, downloads/name))
        natural[split] = natural_candidates(values["Questions"], values["Annotations"], values["Complementary_Pairs"], tok, contractions)
    sets = {}
    sets["natural_test"] = select_natural(natural["val"], {"yes/no":36,"number":24,"other":36}, rng, unique_images=True)
    reserved = pair_images(sets["natural_test"])
    sets["natural_dev"] = select_natural(natural["val"], {"yes/no":12,"number":8,"other":12}, rng, reserved, unique_images=True)
    reserved |= pair_images(sets["natural_dev"])
    sets["natural_train"] = select_natural(natural["train"], {"yes/no":640,"number":256,"other":640}, rng)
    val_ids = [x["id"] for x in instances["val"]["images"]]
    val_pools = object_pools(instances["val"], val_ids)
    for name, count in (("grounding_test",64), ("grounding_dev",24)):
        pairs = []
        for i in range(count):
            pair = draw_grounding(val_pools, "presence" if i<count//2 else "counting", rng,
                                  pair_id=f"{name}-{i:04d}", excluded=reserved)
            reserved |= pair_images([pair])
            pairs.append(pair)
        sets[name] = pairs
    caption_examples = {}
    selected_caption_ids = {}
    for split, count, per_image in (("train",6000,3), ("val",96,1)):
        by_image = defaultdict(list)
        for ann in sorted(captions[split]["annotations"], key=lambda a: a["id"]):
            if len(tok.encode(ann["caption"], add_special_tokens=False)) <= 64:
                by_image[ann["image_id"]].append(ann)
        eligible = sorted(i for i, anns in by_image.items() if len(anns)>=per_image and (split=="train" or i not in reserved))
        if len(eligible)<count:
            raise ValueError("Insufficient caption images")
        chosen = rng.choice(eligible, count, replace=False).tolist()
        selected_caption_ids[split] = chosen
        caption_examples[split] = [{"image_id": i, "caption_id": a["id"], "question": "Describe this image.",
            "answer": a["caption"], "type": "caption"} for i in chosen for a in by_image[i][:per_image]]
    pools = object_pools(instances["train"], selected_caption_ids["train"])
    # Development and test image controls are fixed once, before any generation.
    mappings = {}
    for name in ("grounding_dev", "grounding_test", "natural_dev", "natural_test"):
        pairs = sets[name]
        mapping = {}
        for pair in pairs:
            own = pair_images([pair])
            candidates = sorted(pair_images(pairs)-own)
            for example in pair["examples"]:
                mapping[str(example["question_id"])] = int(rng.choice(candidates))
        mappings[name] = mapping
    isolation = {name: pair_images(pairs) for name, pairs in sets.items() if not name.endswith("train")}
    isolation["caption_dev"] = set(selected_caption_ids["val"])
    for name, ids in isolation.items():
        for other, other_ids in isolation.items():
            if name<other and ids & other_ids:
                raise ValueError(f"Evaluation image leakage between {name} and {other}")
    train_images = set(selected_caption_ids["train"]) | pair_images(sets["natural_train"])
    eval_images = set().union(*isolation.values())
    if train_images & eval_images:
        raise ValueError("COCO train/validation image IDs unexpectedly overlap")
    needed = {"train": train_images, "val": eval_images}
    image_index = {}
    for split in ("train", "val"):
        by_id = {x["id"]: x for x in instances[split]["images"]}
        for image_id in sorted(needed[split]):
            filename = by_id[image_id]["file_name"]
            image_index[str(image_id)] = {"split": split, "file": f"{split}2014/{filename}"}
    # Only selected images are fetched; no 20-GB image archives are needed.
    # The public COCO image endpoints support independently resumable downloads.
    from concurrent.futures import ThreadPoolExecutor
    entries=list(image_index.values())
    def fetch(info):
        from PIL import Image
        path=download(COCO_BASE+info["file"],d/"images"/info["file"])
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return path
    with ThreadPoolExecutor(max_workers=8) as pool:
        for offset in range(0,len(entries),128):
            list(pool.map(fetch,entries[offset:offset+128]))
            # All file handles for this chunk are closed before committing.
            commit()
            print(f"Images {min(offset+128,len(entries))}/{len(entries)}",flush=True)
    write_json(d/"pairs.json", sets)
    write_json(d/"captions.json", caption_examples)
    write_json(d/"grounding_pools.json", pools)
    write_json(d/"image_index.json", image_index)
    write_json(d/"mismatched.json", mappings)
    write_json(d/"split_manifest.json", {"seed": SEED, "caption_ids": selected_caption_ids,
        "pair_ids": {k: [p["pair_id"] for p in v] for k,v in sets.items()},
        "evaluation_images": {k: sorted(v) for k,v in isolation.items()},
        "train_images": sorted(train_images), "question_cap":80,"caption_cap":64,"answer_cap":16,
        "label_review_required": True})
    store.finish(d, signature, [d/f for f in ("pairs.json","captions.json","grounding_pools.json",
        "image_index.json","mismatched.json","split_manifest.json","contractions.json")])

