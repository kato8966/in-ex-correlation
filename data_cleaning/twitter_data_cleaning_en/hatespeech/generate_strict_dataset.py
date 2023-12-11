import pandas as pd

import wordlists

for wordlist_name in ['hatespeech_gender_exp', 'hatespeech_race_exp']:
    query = getattr(wordlists, wordlist_name)()
    if wordlist_name == 'hatespeech_gender_exp':
        targets = ['male', 'female']
    else:
        targets = ['w', 'aa']
    for i, target in enumerate(targets):
        df = pd.read_csv(f'hate_test_{target}_processed.tsv', sep='\t')
        select = [any(word in query.target_sets[i] for word in tweet.split())
                  for tweet in df['Tweet text']]
        strict_df = df[select]
        strict_df.to_csv(f'hate_test_{target}_processed_strict.tsv', '\t',
                         index=False)
