import pandas as pd

import wordlists

for wordlist_name in ['hatespeech_gender_exp', 'hatespeech_race_exp']:
    query = getattr(wordlists, wordlist_name)()
    if wordlist_name == 'hatespeech_gender_exp':
        targets = ['male', 'female']
    else:
        targets = ['w', 'aa']
    for i, target in enumerate(targets):
        tweets = pd.read_csv(f'hate_test_{target}_processed.tsv',
                             sep='\t')['Tweet text']
        cnt = 0
        for tweet in tweets:
            for word in tweet.split():
                if word in query.target_sets[i]:
                    cnt += 1
                    break
        print(f'{wordlist_name} {target}: {cnt} tweets')
