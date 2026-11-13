import pandas as pd
df = pd.read_parquet('main/task3-offline/summary.parquet')
df_eval = df[df['split'] == 'evaluation']

for dset in ['math', 'mathvista']:
    print(f"\n==================== {dset.upper()} ====================")
    sub = df_eval[df_eval['dataset'] == dset]
    methods = sub['method'].unique()
    for m in methods:
        mrow = sub[sub['method'] == m].sort_values('N')
        accs = [f"N={r['N']}: {r['accuracy']*100:.2f}%" for _, r in mrow.iterrows()]
        last = mrow.iloc[-1]
        print(f"{m:18} | {' | '.join(accs)} | SampFail: {last['sampling_failure']*100:.2f}% | SelFail: {last['selection_failure']*100:.2f}%")
