import pandas as pd

data = pd.read_csv('hatespeech_text_label_vote_RESTRICTED_100K.csv', sep='\t',
                   usecols=['Tweet text', 'Label'],
                   converters={'Label': lambda label: int(label in {'abusive', 'hateful'})})  # noqa: E501

for section in ['train', 'val']:
    with open(f'hate_{section}_id.txt') as ids:
        ids = list(map(int, ids))
        section_data = data.iloc[ids]
        section_data.to_csv(f'hate_{section}.tsv', '\t', index=False)

gender_ids = pd.read_csv('hate_test_gender_id.csv')
for gender in ['male', 'female', 'neutral']:
    ids = list(gender_ids['id'].loc[gender_ids['gender'] == gender])
    test_data = data.iloc[ids]
    test_data.to_csv(f'hate_test_{gender}.tsv', '\t', index=False)
