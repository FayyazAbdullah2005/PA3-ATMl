"""Pure scoring routines; selectors never receive correctness labels."""
from __future__ import annotations
import ast
from collections import Counter
from functools import lru_cache
import re
import numpy as np

NUMBER_WORDS = dict(zip(("none", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"),
                       ("0", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10")))


def extract_contractions(source: str):
    match = re.search(r"self\.contractions\s*=\s*(\{.*?\})", source, re.S)
    if not match:
        raise ValueError("Official VQA contraction dictionary not found")
    result = ast.literal_eval(match.group(1).replace("\\\n", ""))
    if len(result) < 90:
        raise ValueError("Incomplete VQA contraction dictionary")
    return {k.lower(): v.lower() for k, v in result.items()}


def vqa_normalize(answer, contractions=None):
    text = str(answer or "").replace("\n", " ").replace("\t", " ").strip().lower()
    original = text
    comma = re.search(r"\d,\d", original) is not None
    for p in (';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!'):
        text = text.replace(p, "" if (p+" " in original or " "+p in original or comma) else " ")
    text = re.sub(r"(?!<=\d)(\.)(?!\d)", "", text)
    words = [NUMBER_WORDS.get(word, word) for word in text.split() if word not in {"a", "an", "the"}]
    contractions = contractions or {}
    return " ".join(contractions.get(word, word) for word in words)


def vqa_soft(prediction, references, contractions=None):
    if len(references) != 10:
        raise ValueError("VQA requires exactly ten human answers")
    pred = vqa_normalize(prediction, contractions)
    refs = [vqa_normalize(x, contractions) for x in references]
    if not pred:
        return 0.0
    return float(np.mean([min(1.0, sum(pred == refs[j] for j in range(10) if j != i)/3) for i in range(10)]))


def last_boxed(text):
    start = str(text or "").rfind("\\boxed{")
    if start < 0:
        return None
    pos, depth = start+7, 1
    begin = pos
    while pos < len(text):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:pos].strip() or None
        pos += 1
    return None


@lru_cache(maxsize=100000)
def parsed_math(text):
    import threading,os
    if os.name=="nt" or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Bounded Math-Verify scoring must run in a Linux main process. Use the Modal CPU commands.")
    from math_verify import parse, LatexExtractionConfig, ExprExtractionConfig
    return parse("$"+str(text).strip().strip("$")+"$",
                 extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])


@lru_cache(maxsize=100000)
def math_equivalent(a, b):
    if a is None or b is None or not str(a).strip() or not str(b).strip():
        return False
    if str(a).strip() == str(b).strip():
        return True
    from math_verify import verify
    aa, bb = parsed_math(a), parsed_math(b)
    return bool(aa and bb and verify(aa, bb) and verify(bb, aa))


def mathvista_normalize(answer, metadata):
    if answer is None or not str(answer).strip():
        return None
    answer = str(answer).strip()
    if metadata.get("question_type") == "multi_choice":
        choices = metadata["choices"]
        letters = [chr(65+i) for i in range(len(choices))]
        cleaned = answer.strip().strip("(). ").upper()
        if cleaned in letters:
            return cleaned
        matches = [i for i, choice in enumerate(choices) if str(choice).strip().casefold() == answer.casefold()]
        if len(matches) == 1:
            return letters[matches[0]]
        # MathVista's released normalization chooses the closest option by edit
        # distance when the extracted answer is not an option letter/value.
        def distance(a,b):
            previous=list(range(len(b)+1))
            for i,aa in enumerate(a,1):
                current=[i]
                for j,bb in enumerate(b,1):
                    current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(aa!=bb)))
                previous=current
            return previous[-1]
        best=min(range(len(choices)),key=lambda i:distance(answer,str(choices[i])))
        return letters[best]
    kind = metadata.get("answer_type", "text")
    try:
        if kind == "integer":
            return str(int(float(answer)))
        if kind == "float":
            precision = int(metadata["precision"])
            return str(round(float(answer), precision))
        if kind == "list":
            value = ast.literal_eval(answer)
            return str(value) if isinstance(value, list) else answer
    except (ValueError, TypeError, OverflowError, SyntaxError):
        return answer
    return answer


def group_answers(answers, equivalent):
    """Prefix-local representative grouping in the current rollout order."""
    representatives, groups = [], []
    for answer in answers:
        missing = answer is None or not str(answer).strip()
        found = None
        for i, representative in enumerate(representatives):
            if (missing and representative is None) or (not missing and representative is not None and equivalent(answer, representative)):
                found = i
                break
        if found is None:
            found = len(representatives)
            representatives.append(None if missing else answer)
        groups.append(found)
    return groups, representatives


def mean_scores(rollouts, vision=False):
    from scipy.stats import rankdata
    scores = {"logp": np.array([np.mean(r["logp"]) for r in rollouts], dtype=float),
              "certainty": np.array([np.mean(r["kl_u_p"]) for r in rollouts], dtype=float)}
    if vision:
        pooled = np.concatenate([r["kl_u_p"] for r in rollouts])
        low, high = np.quantile(pooled, [.05, .95], method="linear")
        calibrated, engagement, token_fusion, correlations = [], [], [], []
        for r in rollouts:
            c = np.clip((np.asarray(r["kl_u_p"])-low)/max(high-low, 1e-8), 0, 1)
            image = np.asarray(r["img_attn_total"], dtype=float)
            denominator = image+np.asarray(r["prompt_text_attn_total"])+np.asarray(r["generated_text_attn_total"])
            if np.any(denominator <= 0):
                raise ValueError("Visual-attention denominator must be positive")
            v = image/denominator
            engagement.append(float(v.mean()))
            calibrated.append(float(c.mean()))
            token_fusion.append(float((c*v).mean()))
            correlations.append(float(np.corrcoef(c, v)[0, 1]) if np.std(c)>0 and np.std(v)>0 else None)
        scores["vision"] = np.array(engagement)
        scores["visconf"] = scores["vision"]*np.array(calibrated)
        scores["token_fusion"] = np.array(token_fusion)
        scores["confidence_vision_correlation"] = correlations
    return scores


def vote(groups, weights=None):
    weights = np.ones(len(groups)) if weights is None else np.asarray(weights, dtype=float)
    if len(groups) != len(weights) or not len(groups):
        raise ValueError("Nonempty groups and matching weights required")
    totals = {}
    first = {}
    for i, (group, weight) in enumerate(zip(groups, weights)):
        totals[group] = totals.get(group, 0.0)+float(weight)
        first.setdefault(group, i)
    winner = min(totals, key=lambda g: (-totals[g], first[g]))
    ranked = sorted(totals.values(), reverse=True)
    margin = (ranked[0]-(ranked[1] if len(ranked)>1 else 0))/max(float(weights.sum()), 1e-12)
    unique = sum(np.isclose(v, ranked[0], rtol=0, atol=1e-12) for v in totals.values()) == 1
    return winner, margin, unique, totals


def rwc_weights(scores):
    from scipy.stats import rankdata
    scores = np.asarray(scores, dtype=float)
    if scores.size == 1:
        return np.ones(1)
    if not np.isfinite(scores).all():
        raise ValueError("Quality scores must be finite")
    ranks = rankdata(-scores, method="average")
    return (len(scores)-ranks)/(len(scores)-1)


def select(groups, scores, method):
    if method == "first":
        return 0
    if method == "sc":
        winner = vote(groups)[0]
    elif method.endswith("_rwc"):
        winner = vote(groups, rwc_weights(scores[method[:-4]]))[0]
    else:
        return int(np.argmax(scores[method]))
    return groups.index(winner)


def bootstrap_difference(a, b, count=1000, seed=6304):
    difference = np.asarray(a, dtype=float)-np.asarray(b, dtype=float)
    if difference.ndim != 1 or not len(difference):
        raise ValueError("Bootstrap expects nonempty matched question/pair vectors")
    rng = np.random.default_rng(seed)
    values = np.array([difference[rng.integers(len(difference), size=len(difference))].mean() for _ in range(count)])
    return {"difference": float(difference.mean()), "low": float(np.quantile(values, .025)),
            "high": float(np.quantile(values, .975)), "resamples": count}

