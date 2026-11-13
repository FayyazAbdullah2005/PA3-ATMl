import pandas as pd
import json

df = pd.read_csv('main/task1_responses.csv')
for pid in ['what_is_your_wish', 'enough_of_my_husband', 'what_is_your_wish_json', 'what_is_your_wish_template']:
    print(f'=== PROMPT: {pid} ===')
    sub = df[df['prompt_id'] == pid][['condition', 'generated_text']]
    for _, r in sub.iterrows():
        print(f"[{r['condition']}]: {r['generated_text']}")
