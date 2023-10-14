from collections import Counter
from itertools import chain
from math import log2

import pandas as pd

K = 16  # extract K words
thres_count = 10


def count_words(tweets, normalize=False):
    words = chain.from_iterable(tweet.split() for tweet in tweets)
    return pd.Series(words).value_counts(normalize=normalize)


def estimate_paras(df):
    labels = df['Label']
    mu = labels.value_counts(normalize=True)[1]
    theta = []
    for label in range(2):
        tweets_labeled = df['Tweet text'].loc[labels == label]
        theta.append(count_words(tweets_labeled, True))
    return mu, theta


def extract(mu, theta, word_counts, tweet_counts):
    extracted_words = []

    def cond_p(word, label):
        # conditional probability that word w appears given label l
        if word in theta[label].index:
            return theta[label][word]
        else:
            return 0.0

    for label in range(2):
        pmis = [(log2(cond_p(word, label)) - log2((1.0 - mu) * cond_p(word, 0)
                                                    + mu * cond_p(word, 1)),
                 word_counts[word],
                 word) for word in theta[label].index
                       if tweet_counts[word] >= thres_count]
        pmis.sort(reverse=True)
        assert len(pmis) >= K
        _, _, ext_words = zip(*pmis[:K])
        extracted_words.append(ext_words)

    return extracted_words


if __name__ == '__main__':
    hate_test = {gender: pd.read_csv(f'hate_test_{gender}_processed.tsv', sep='\t')
                 for gender in ['male', 'female']}

    hate_test_concat = pd.concat(hate_test.values())
    word_counts = count_words(hate_test_concat['Tweet text'])
    tweet_counts = Counter()
    for tweet in hate_test_concat['Tweet text']:
        words = set(tweet.split())
        tweet_counts.update(words)
    mu, theta = estimate_paras(hate_test_concat)
    extracted_words = extract(mu, theta, word_counts, tweet_counts)
    for category, words in zip(['non hate speech', 'hate speech'], extracted_words):
        print(f'{category} words')
        print(words)
    print()

    hate_test['male']['Label'] = pd.Series([1] * len(hate_test['male']))
    hate_test['female']['Label'] = pd.Series([0] * len(hate_test['female']))

    mu, theta = estimate_paras(pd.concat(hate_test.values()))
    extracted_words = extract(mu, theta, word_counts, tweet_counts)
    for category, words in zip(['female', 'male'], extracted_words):
        print(f'{category} words')
        print(words)
