import os
import pandas as pd
import json
import numpy as np

# Distinct answers per N
df3 = pd.read_parquet('main/task3-offline/summary.parquet')
df3_eval = df3[df3['split'] == 'evaluation']

print("=== DISTINCT ANSWERS PER N ===")
piv_dist = df3_eval.pivot_table(index='dataset', columns='N', values='distinct_answers', aggfunc='mean')
print(piv_dist.to_string())

# Signal summary
print("\n=== TASK 3 SIGNAL SUMMARY ===")
if os.path.exists('main/task3_signal_summary.csv'):
    sig_df = pd.read_csv('main/task3_signal_summary.csv')
    print(sig_df.to_string())

# Adaptive stopping summary
print("\n=== TASK 3 ADAPTIVE SUMMARY ===")
with open('main/task3-adaptive/summary.json') as f:
    ad_sum = json.load(f)
print(json.dumps(ad_sum, indent=2))

# Stopping distribution
print("\n=== TASK 3 STOPPING DISTRIBUTION ===")
with open('main/task3-adaptive/stopping_distribution.json') as f:
    ad_dist = json.load(f)
print(json.dumps(ad_dist, indent=2))
