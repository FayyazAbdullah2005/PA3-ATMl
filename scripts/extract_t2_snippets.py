import pandas as pd

df2 = pd.read_csv('main/task2-evaluation/196/examples.csv')
sub_g = df2[df2['suite'] == 'grounding_test']
print("\nSample grounding pairs:")
print(sub_g[['pair_id', 'condition', 'type', 'question', 'answer', 'prediction', 'score', 'partner_match']].head(12).to_string())

sub_n = df2[df2['suite'] == 'natural_test']
print("\nSample natural pairs:")
print(sub_n[['pair_id', 'condition', 'type', 'question', 'answer', 'prediction', 'score', 'partner_match']].head(12).to_string())
