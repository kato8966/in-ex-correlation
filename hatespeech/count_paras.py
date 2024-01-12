from concurrent.futures import ProcessPoolExecutor
import json
from os import path
from statistics import variance

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch.distributions import Bernoulli
from torch.linalg import vector_norm
from torch import nn
from torch.nn import functional as NNF
from torch.utils.data import DataLoader, Dataset
from torchtext import functional as TTF

GPUS = 8


class HatespeechDataset(Dataset):
    def __init__(self, data, voc):
        self.tweets = []
        for tweet in data['Tweet text']:
            tweet = tweet.split()
            tweet = [voc[word] if word in voc else voc['<UNK>'] for word in tweet]
            self.tweets.append(tweet)
        self.labels = data['Label']

    def __getitem__(self, idx):
        return self.tweets[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


def collate_fn(data):
    tweets, labels = zip(*data)
    return TTF.to_tensor(list(tweets), 0), torch.tensor(labels)


class MyDropout(nn.Module):
    def __init__(self, p=0.5, device='cpu'):
        super().__init__()
        self.bernoulli = Bernoulli(torch.tensor(p, device=device))

    def forward(self, x):
        if self.training:
            mask = self.bernoulli.sample(x.shape)
            x *= mask
        else:
            x *= self.bernoulli.probs
        return x


class Detector(nn.Module):
    def __init__(self, word_emb, device):
        super().__init__()
        self.word_emb = nn.Embedding.from_pretrained(word_emb)
        self.conv1 = nn.Conv1d(300, 100, 3)
        self.conv2 = nn.Conv1d(300, 100, 4)
        self.conv3 = nn.Conv1d(300, 100, 5)
        self.relu = nn.ReLU()
        self.pool = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten())
        self.dropout = MyDropout(device=device)
        self.linear = nn.Linear(300, 2)
        self.to(device)

    def forward(self, tweets):
        tweets = NNF.pad(tweets, (0, max(5 - tweets.shape[1], 0)), value=0)
        X = self.word_emb(tweets)
        X = torch.transpose(X, 1, 2)
        Z1 = self.pool(self.relu(self.conv1(X)))
        Z2 = self.pool(self.relu(self.conv2(X)))
        Z3 = self.pool(self.relu(self.conv3(X)))
        Z = torch.cat((Z1, Z2, Z3), dim=1)
        Z = self.dropout(Z)
        return self.linear(Z)

    def predict(self, tweets):
        logits = self(tweets)
        return torch.argmax(logits, 1)


def main(args, random_seed, gpu_id):
    torch.manual_seed(random_seed)
    rng = np.random.default_rng(random_seed)
    if args['bias_modification'] == 'none':
        id = f"{args['word_emb']}_original{args['id']}"
        word_emb_file = '../w2v/vectors/twitter.txt'
    elif args['bias_modification'] == 'db':
        id = f"{args['word_emb']}_db_{args['wordlist']}_{args['bias_type']}"\
             f"_{args['sample_prob']}"
        word_emb_file = f'../w2v/vectors/twitter_{id.replace(args["word_emb"] + "_", "")}.txt'
    else:
        assert args['bias_modification'] == 'ar'
        id = f"{args['word_emb']}_ar_{args['wordlist']}_{args['bias_type']}_reg{args['reg']}"\
             f"_sim{args['sim']}_ant{args['ant']}"
        word_emb_file = '../attract-repel/vectors/'\
                        f'twitter_{id.replace("_ar_", "_")}.txt'
    if 'wordlist' in args:
        if 'gender' in args['wordlist']:
            targets = ['male', 'female']
        else:
            assert 'race' in args['wordlist']
            targets = ['w', 'aa']
    else:
        targets = ['male', 'female', 'w', 'aa']
    device = f'cuda:{gpu_id}'

    train_data = pd.read_csv(path.join('..', 'data_cleaning',
                                       'twitter_data_cleaning_en',
                                       'hatespeech',
                                       'hate_train_processed.tsv'),
                             sep='\t')
    val_data = pd.read_csv(path.join('..', 'data_cleaning',
                                     'twitter_data_cleaning_en',
                                     'hatespeech',
                                     'hate_val_processed.tsv'),
                           sep='\t')
    test_data = pd.read_csv(path.join('..', 'data_cleaning',
                                      'twitter_data_cleaning_en',
                                      'hatespeech',
                                      'hate_test_processed.tsv'),
                            sep='\t')

    all_voc = {'<PAD>'}
    for data in [train_data, val_data, test_data]:
        for tweet in data['Tweet text']:
            all_voc.update(tweet.split())

    voc = {'<PAD>': 0}
    word_emb = [np.zeros(300)]
    elements = []
    with open(word_emb_file) as fin:
        for line in fin:
            line = line.split(' ', 1)
            word = line[0]
            if word in all_voc:
                voc[word] = len(voc)
                vector = np.fromstring(line[1], sep=' ')
                word_emb.append(vector)
                elements += vector.tolist()

    var = variance(elements)
    interval = (3 * var) ** 0.5
    for tweet in train_data['Tweet text']:
        for word in tweet.split():
            if word not in voc:
                voc[word] = len(voc)
                word_emb.append(rng.uniform(-interval, interval, 300))

    word_emb = torch.from_numpy(np.array(word_emb, dtype=np.float32))

    model = Detector(word_emb, device)

    print(sum(torch.numel(p) for p in model.parameters() if p.requires_grad))

if __name__ == '__main__':
    torch.use_deterministic_algorithms(True, warn_only=True)
    args = ([{'bias_modification': 'none', 'word_emb': word_emb, 'id': i}
             for word_emb in ['w2v', 'ft'] for i in [1]])
    with open('random_seeds.json') as fin:
        random_seeds = json.load(fin)
    main(args[0], random_seeds[0], 0)
