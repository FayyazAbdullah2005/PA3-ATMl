import json
import pandas as pd

print("=== ACTUAL TASK 2 (196 TOKENS) SUMMARY.JSON ===")
with open('main/task2-evaluation/196/statistics/summary.json') as f:
    t2_196 = json.load(f)

for row in t2_196:
    print(f"Suite: {row['suite']:15} | Cond: {row['condition']:10} | Type: {row['type']:8} | Acc: {row['accuracy']:.3f} | PairAcc: {row['pair_accuracy']:.3f} | Ident: {row['identical_answer_rate']:.3f} | PartMatch: {row['partner_match_rate']:.3f} | Empty: {row['empty_answer_rate']:.3f} | NLL: {row['reference_nll']:.3f} | Latency: {row['seconds_per_answer']*1000:.1f}ms")

print("\n=== ACTUAL TASK 2 (196 TOKENS) ADVANTAGES ===")
with open('main/task2-evaluation/196/statistics/correct_image_advantages.json') as f:
    t2_196_adv = json.load(f)

for row in t2_196_adv:
    print(f"Suite: {row['suite']:15} | Type: {row['type']:8} | Metric: {row['metric']:5} | Diff: {row['difference']:.3f} [{row['low']:.3f}, {row['high']:.3f}]")

print("\n=== ACTUAL TASK 2 (49 TOKENS) SUMMARY.JSON ===")
with open('main/task2-evaluation/49/statistics/summary.json') as f:
    t2_49 = json.load(f)

for row in t2_49:
    print(f"Suite: {row['suite']:15} | Cond: {row['condition']:10} | Type: {row['type']:8} | Acc: {row['accuracy']:.3f} | PairAcc: {row['pair_accuracy']:.3f} | Ident: {row['identical_answer_rate']:.3f} | PartMatch: {row['partner_match_rate']:.3f} | Empty: {row['empty_answer_rate']:.3f} | NLL: {row['reference_nll']:.3f} | Latency: {row['seconds_per_answer']*1000:.1f}ms")

print("\n=== ACTUAL TASK 2 (49 TOKENS) ADVANTAGES ===")
with open('main/task2-evaluation/49/statistics/correct_image_advantages.json') as f:
    t2_49_adv = json.load(f)

for row in t2_49_adv:
    print(f"Suite: {row['suite']:15} | Type: {row['type']:8} | Metric: {row['metric']:5} | Diff: {row['difference']:.3f} [{row['low']:.3f}, {row['high']:.3f}]")
