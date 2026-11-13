import json

def get_row(data, suite, cond, stype):
    for r in data:
        if r['suite'] == suite and r['condition'] == cond and r['type'] == stype:
            return r
    return None

def get_adv(data, suite, stype, metric):
    for r in data:
        if r['suite'] == suite and r['type'] == stype and r['metric'] == metric:
            return r
    return None

with open('main/task2-evaluation/196/statistics/summary.json') as f:
    s196 = json.load(f)
with open('main/task2-evaluation/196/statistics/correct_image_advantages.json') as f:
    a196 = json.load(f)

print("=== TABLE 3 (196 TOKENS) REAL VALUES ===")
configs = [
    ('grounding_test', 'correct', 'all'),
    ('grounding_test', 'swap', 'all'),
    ('grounding_test', 'mismatched', 'all'),
    ('grounding_test', 'none', 'all'),
    ('grounding_test', 'correct', 'presence'),
    ('grounding_test', 'correct', 'counting'),
    ('natural_test', 'correct', 'all'),
    ('natural_test', 'swap', 'all'),
    ('natural_test', 'mismatched', 'all'),
    ('natural_test', 'none', 'all'),
    ('natural_test', 'correct', 'yes/no'),
    ('natural_test', 'correct', 'number'),
    ('natural_test', 'correct', 'other'),
]

for suite, cond, stype in configs:
    r = get_row(s196, suite, cond, stype)
    adv_score = get_adv(a196, suite, stype, 'score') if cond == 'correct' else None
    adv_nll = get_adv(a196, suite, stype, 'nll') if cond == 'correct' else None
    acc = f"{r['accuracy']:.3f}"
    pair = f"{r['pair_accuracy']:.3f}"
    ident = f"{r['identical_answer_rate']:.3f}"
    part = f"{r['partner_match_rate']:.3f}"
    empty = f"{r['empty_answer_rate']:.2f}"
    nll = f"{r['reference_nll']:.3f}"
    time = f"{r['seconds_per_answer']*1000:.1f}"
    d_score = f"+{adv_score['difference']:.3f} [{adv_score['low']:.3f}, {adv_score['high']:.3f}]" if adv_score else "--"
    d_nll = f"+{adv_nll['difference']:.3f} [{adv_nll['low']:.3f}, {adv_nll['high']:.3f}]" if adv_nll else "--"
    print(f"{suite:15} | {cond:10} | {stype:8} | Acc:{acc} | Pair:{pair} | Ident:{ident} | Part:{part} | Empty:{empty} | NLL:{nll} | Time:{time} | {d_score} | {d_nll}")

with open('main/task2-evaluation/49/statistics/summary.json') as f:
    s49 = json.load(f)
with open('main/task2-evaluation/49/statistics/correct_image_advantages.json') as f:
    a49 = json.load(f)

print("\n=== APPENDIX TABLE (49 TOKENS) REAL VALUES ===")
configs_49 = [
    ('grounding_test', 'correct', 'all'),
    ('grounding_test', 'swap', 'all'),
    ('grounding_test', 'mismatched', 'all'),
    ('grounding_test', 'none', 'all'),
    ('natural_test', 'correct', 'all'),
    ('natural_test', 'swap', 'all'),
    ('natural_test', 'mismatched', 'all'),
    ('natural_test', 'none', 'all'),
]
for suite, cond, stype in configs_49:
    r = get_row(s49, suite, cond, stype)
    adv_score = get_adv(a49, suite, stype, 'score') if cond == 'correct' else None
    adv_nll = get_adv(a49, suite, stype, 'nll') if cond == 'correct' else None
    acc = f"{r['accuracy']:.3f}"
    pair = f"{r['pair_accuracy']:.3f}"
    ident = f"{r['identical_answer_rate']:.3f}"
    part = f"{r['partner_match_rate']:.3f}"
    nll = f"{r['reference_nll']:.3f}"
    d_score = f"+{adv_score['difference']:.3f}" if adv_score else "--"
    d_nll = f"+{adv_nll['difference']:.3f}" if adv_nll else "--"
    print(f"{suite:15} | {cond:10} | {stype:8} | Acc:{acc} | Pair:{pair} | Ident:{ident} | Part:{part} | NLL:{nll} | {d_score} | {d_nll}")
