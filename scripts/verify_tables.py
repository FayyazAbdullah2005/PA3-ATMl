import pandas as pd
import json

print("=== 1. TASK 1 BENCHMARKS ===")
df1 = pd.read_csv('main/task1_benchmarks.csv')
for _, r in df1.iterrows():
    print(f"{r['condition']:10} | GSM8K: {r['gsm8k_cot']:.2f} ({r['gsm8k_cot_delta_pp']:+.2f}) | Hella: {r['hellaswag']:.2f}/{r['hellaswag_raw_accuracy']:.2f} ({r['hellaswag_delta_pp']:+.2f}) | BoolQ: {r['boolq']:.2f} ({r['boolq_delta_pp']:+.2f}) | ARC: {r['arc_easy']:.2f}/{r['arc_easy_raw_accuracy']:.2f} ({r['arc_easy_delta_pp']:+.2f}) | Trunc: {r['gsm8k_truncated_count']} ({r['gsm8k_truncation_rate']:.2f}%) | Unext: {r['gsm8k_unextractable_count']}")

print("\n=== 2. TASK 1 RESOURCES ===")
df1_res = pd.read_csv('main/task1_resources.csv')
for _, r in df1_res.iterrows():
    print(f"{r['method']:10} | Params: {r['trainable_parameters']/1e6:.2f}M | VRAM: {r['peak_gpu_bytes']/(1024**3):.2f} GB | Time: {r['training_wall_seconds']/60:.1f} min | Ckpt: {r['single_checkpoint_bytes']/(1024**3):.2f} GB | L2: {r['l2_displacement']} | L2_mid: {r['l2_displacement_midpoint']}")

print("\n=== 3. TASK 1 DIAGNOSTICS L13 ===")
df1_diag = pd.read_csv('main/task1_diagnostics_summary.csv')
print(df1_diag[df1_diag['layer'] == 13][['condition', 'group', 'rho', 'eta', 'R_mean']])

print("\n=== 4. TASK 2 RESOURCES ===")
df2_res = pd.read_csv('main/task2_resources.csv')
print(df2_res.to_string())

print("\n=== 5. TASK 2 EVALUATION 196 ===")
with open('main/task2-evaluation/196/statistics/summary.json') as f:
    d2 = json.load(f)
for row in d2:
    print(f"{row['suite']:15} | {row['condition']:10} | {row['type']:8} | Acc: {row['accuracy']:.3f} | PairAcc: {row['pair_accuracy']:.3f} | Ident: {row['identical_answer_rate']:.3f} | PartMatch: {row['partner_match_rate']:.3f} | Empty: {row['empty_answer_rate']:.3f} | NLL: {row['reference_nll']:.3f} | Time: {row['seconds_per_answer']*1000:.1f}ms")

print("\n=== 6. TASK 2 ADVANTAGES 196 ===")
with open('main/task2-evaluation/196/statistics/correct_image_advantages.json') as f:
    d2_adv = json.load(f)
for row in d2_adv:
    print(f"{row['suite']:15} | {row['type']:8} | {row['metric']:5} | Diff: {row['difference']:.3f} [{row['low']:.3f}, {row['high']:.3f}]")

print("\n=== 7. TASK 2 EVALUATION 49 ===")
with open('main/task2-evaluation/49/statistics/summary.json') as f:
    d2_49 = json.load(f)
for row in d2_49:
    print(f"{row['suite']:15} | {row['condition']:10} | {row['type']:8} | Acc: {row['accuracy']:.3f} | PairAcc: {row['pair_accuracy']:.3f} | Ident: {row['identical_answer_rate']:.3f} | PartMatch: {row['partner_match_rate']:.3f} | Empty: {row['empty_answer_rate']:.3f} | NLL: {row['reference_nll']:.3f} | Time: {row['seconds_per_answer']*1000:.1f}ms")

print("\n=== 8. TASK 2 ADVANTAGES 49 ===")
with open('main/task2-evaluation/49/statistics/correct_image_advantages.json') as f:
    d2_49_adv = json.load(f)
for row in d2_49_adv:
    print(f"{row['suite']:15} | {row['type']:8} | {row['metric']:5} | Diff: {row['difference']:.3f} [{row['low']:.3f}, {row['high']:.3f}]")

print("\n=== 9. TASK 3 OFFLINE EVALUATION ===")
df3_off = pd.read_parquet('main/task3-offline/summary.parquet')
df3_eval = df3_off[df3_off['split'] == 'evaluation']
for dset in ['math', 'mathvista']:
    print(f"*** {dset} ***")
    sub = df3_eval[df3_eval['dataset'] == dset]
    piv = sub.pivot(index='method', columns='N', values='accuracy')
    print(piv.to_string())

print("\n=== 10. TASK 3 LIVE SUMMARY ===")
with open('main/task3-live/summary.json') as f:
    print(json.dumps(json.load(f), indent=2))

print("\n=== 11. TASK 3 ADAPTIVE SUMMARY ===")
with open('main/task3-adaptive/summary.json') as f:
    print(json.dumps(json.load(f), indent=2))

print("\n=== 12. TASK 3 ADAPTIVE STOPPING DISTRIBUTION ===")
with open('main/task3-adaptive/stopping_distribution.json') as f:
    print(json.dumps(json.load(f), indent=2))
