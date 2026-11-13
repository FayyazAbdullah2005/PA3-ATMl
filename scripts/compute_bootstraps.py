import pandas as pd
import numpy as np

df = pd.read_csv('main/task3-offline/question_means.csv', low_memory=False)
df_eval = df[(df['split'] == 'evaluation') & (df['N'] == 32)]

def paired_bootstrap_ci(df_data, dset, m1, m2, n_boot=1000, seed=6304):
    np.random.seed(seed)
    sub1 = df_data[(df_data['dataset'] == dset) & (df_data['method'] == m1)].sort_values('sample_id')
    sub2 = df_data[(df_data['dataset'] == dset) & (df_data['method'] == m2)].sort_values('sample_id')
    
    y1 = sub1['correct'].values
    y2 = sub2['correct'].values
    assert len(y1) == len(y2), f"Lengths differ: {len(y1)} vs {len(y2)}"
    
    diff = y1 - y2
    mean_diff = np.mean(diff)
    
    boot_diffs = []
    n = len(diff)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        boot_diffs.append(np.mean(diff[idx]))
        
    low = np.percentile(boot_diffs, 2.5)
    high = np.percentile(boot_diffs, 97.5)
    return mean_diff, low, high

comparisons = [
    ('math', 'certainty_rwc', 'sc', 'MATH: Self-Certainty+RWC vs SC'),
    ('math', 'certainty', 'logp', 'MATH: Self-Certainty vs Logp'),
    ('mathvista', 'visconf_rwc', 'sc', 'MathVista: VisConf+RWC vs SC'),
    ('mathvista', 'visconf', 'logp', 'MathVista: VisConf vs Logp'),
    ('mathvista', 'visconf_rwc', 'certainty_rwc', 'MathVista: VisConf+RWC vs Certainty+RWC'),
]

print("=== 1,000 PAIRED BOOTSTRAP 95% CIs AT N=32 ===")
for dset, m1, m2, desc in comparisons:
    diff, low, high = paired_bootstrap_ci(df_eval, dset, m1, m2)
    print(f"{desc:42} | Diff: {diff*100:+.2f} pp [95% CI: {low*100:+.2f}, {high*100:+.2f}] pp")
