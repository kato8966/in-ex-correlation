# %%
from nltk.tokenize import TweetTokenizer
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
with open('stream_2017_04_cleaned.txt', 'w') as gg:
    for txt in ff:
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
        gg.write(txt+'\n')

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
with open('stream_2017_04_processed.txt', 'w') as gg:
    tweet_tokenizer = TweetTokenizer()

    for tweet in ff:
        new = ''
        tokens = tweet_tokenizer.tokenize(tweet)
        for token in tokens:
            if token not in vocab_fin:
                new += '<UNK> '
            else:
                new += token+' '
        #print (new)
        gg.write(new+'\n')

# %%
#unnecessary


