import re

import pandas as pd

data = pd.read_csv('hatespeech_text_label_vote_RESTRICTED_100K.csv', sep='\t',
                   usecols=['Tweet text', 'Label'],
                   converters={'Label': lambda label: int(label in {'abusive', 'hateful'})})  # noqa: E501


def convert(text):
    # convert Unicode references in text to its corresponding characters
    text = text.replace('&amp;', '&')
    text = text.replace('&amp;', '&')  # apply &amp; -> & twice so that &amp;amp; (in line 2191) is reduced to & (noqa: E501)
    text = text.replace('&apos;', "'")
    text = text.replace('&quot;', '"')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&pound;', '£')
    text = text.replace('&rsquo;', '’')
    text = text.replace('&THORN;', 'Þ')
    text = text.replace('&nbsp;', ' ')

    def convert_unicode(match):
        code = match.group(1)
        if code[0] == 'x':
            code = int(code[1:], 16)
        else:
            code = int(code)
        return chr(code)

    text = re.sub(r'&#(\d+);', convert_unicode, text, flags=re.ASCII)
    text = re.sub(r'&#(x[a-fA-F0-9]+);', convert_unicode, text)
    return text


data['Tweet text'] = data['Tweet text'].apply(convert)

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
