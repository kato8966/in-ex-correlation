# %%
from collections import Counter
from nltk.tokenize import TweetTokenizer
from os import path
import pandas as pd
import re
#import demoji
#demoji.download_codes()

# %%
'''
This program assumes that your input file contains only tweet content and nothing else
part 1 - removes hashtags, mentions and urls
part 2 - tokenisation to get vocabulary and counts
part 3 - removes low frequency words from vocabulary
part 4 - replaces OOV words by UNK token
'''

# %%
#part 1
with open('stream_2017_04.txt') as fin:
    ff = fin.readlines()


def clean(texts):
    cleaned_texts = []
    for txt in texts:
        if txt == '\n':
            continue
        #print (txt)
        txt = txt.lower()
        #txt = re.sub(r'\#[a-zA-Z0-9]+', " <HASH> ", txt)
        txt = re.sub(r'\#', " <HASH> ", txt)
        txt = re.sub(r'\@\w+', " <MENTION> ", txt, flags=re.ASCII)
        txt = re.sub(r'https?:\/\/\S+', " <URL> ", txt)
        #txt = demoji.replace(txt," <EMOJI> ")
        txt = re.sub(r'\s+'," ",txt)

        #tweet = " ".join(re.split("[^a-zA-Z.,!?]*", txt.lower())).strip()
        #print (tweet)
        #txt = re.sub(r'[[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F1E0-\U0001F1FF]|[\U00002702-\U000027B0]|[\U000024C2-\U0001F251]]+',"<EMOJI>", txt)
        cleaned_texts.append(txt)
    return cleaned_texts


cleaned_ff = clean(ff)
with open('stream_2017_04_cleaned.txt', 'w') as gg:
    for txt in cleaned_ff:
        print(txt, file=gg)

# %%
#part 2
from nltk import ngrams, FreqDist
#all_counts = dict()
with open('stream_2017_04_cleaned.txt') as fin:
    ff = fin.readlines()
#all_counts = FreqDist(ngrams(ff[:9000], 3))
#print (sorted(all_counts))
vocab = dict()
tweet_tokenizer = TweetTokenizer()
for tweet in ff:
    tokens = tweet_tokenizer.tokenize(tweet)
    for token in tokens:
        if token in vocab.keys():
            vocab[token] += 1
        else:
            vocab[token] = 1
#sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=False)
#sorted_d = dict(sorted(vocab.items(), key=operator.itemgetter(1)))
#print (type(vocab))
#print(vocab)


# %%
#part 3
oov=[]
vocab_fin=['<UNK>']
for key in vocab.keys():
    #print (pair)
    if vocab[key]>9:
        vocab_fin.append(key)

print(len(vocab_fin))

# %%
#part 4
def tokenize(tweets):
    tweet_tokenizer = TweetTokenizer()
    tokenized_tweets = []

    for tweet in tweets:
        new = ''
        tokens = tweet_tokenizer.tokenize(tweet)
        for token in tokens:
            if token not in vocab_fin:
                new += '<UNK> '
            else:
                new += token+' '
        #print (new)
        tokenized_tweets.append(new)
    return tokenized_tweets


tokenized_ff = tokenize(ff)
with open('stream_2017_04_processed.txt', 'w') as gg:
    for tweet in tokenized_ff:
        print(tweet, file=gg)

# %%
hate_train = pd.read_csv('hatespeech/hate_train.tsv', sep='\t')
vocab = Counter()
for tweet in hate_train['Tweet text']:
    tokens = tweet_tokenizer.tokenize(tweet)
    for token in tokens:
        if token not in vocab_fin:
            vocab[token] += 1

for word, cnt in vocab.items():
    if cnt > 9:
        vocab_fin.append(word)

for filename in ['hate_train', 'hate_val', 'hate_test_male',
                 'hate_test_female']:
    hatespeech = pd.read_csv(path.join('hatespeech', f'{filename}.tsv'),
                             sep='\t')
    hatespeech['Tweet text'] = pd.Series(tokenize(hatespeech['Tweet text']))
    hatespeech.to_csv(path.join('hatespeech', f'{filename}_processed.tsv'),
                      '\t', index=False)
