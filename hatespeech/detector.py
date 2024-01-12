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
    model_output = f'models/{id}.pt'
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
    bias_test_data = {target: pd.read_csv(path.join('..', 'data_cleaning',
                                                    'twitter_data_cleaning_en',
                                                    'hatespeech',
                                                    f'hate_test_{target}_processed.tsv'),  # noqa: E501
                                          sep='\t')
                      for target in targets}

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

    train_data = HatespeechDataset(train_data, voc)
    val_data = HatespeechDataset(val_data, voc)
    test_data = HatespeechDataset(test_data, voc)
    bias_test_data = {target: HatespeechDataset(bias_test_data[target], voc)
                      for target in targets}

    train_dataloader = DataLoader(train_data, 50, True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, 50, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, 50, collate_fn=collate_fn)
    bias_test_dataloader = {target: DataLoader(bias_test_data[target], 50,
                                               collate_fn=collate_fn)
                            for target in targets}

    model = Detector(word_emb, device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), rho=0.95)
    last_val_loss = float('inf')

    while True:
        model.train()
        for tweet_batch, label_batch in train_dataloader:
            tweet_batch = tweet_batch.to(device)
            label_batch = label_batch.to(device)

            predictions = model(tweet_batch)
            loss = loss_fn(predictions, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for i in range(2):
                    norm = vector_norm(model.linear.weight[i])
                    if norm > 3.0:
                        model.linear.weight[i] *= 3.0 / norm

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tweet_batch, label_batch in val_dataloader:
                tweet_batch = tweet_batch.to(device)
                label_batch = label_batch.to(device)

                predictions = model(tweet_batch)
                loss = loss_fn(predictions, label_batch)
                val_loss += loss.item() * label_batch.shape[0]

        if val_loss >= last_val_loss:
            break
        torch.save(model.state_dict(), model_output)
        last_val_loss = val_loss

    model.load_state_dict(torch.load(model_output))

    def eval_model(dataloader):
        predictions = []
        labels = []

        with torch.no_grad():
            for tweet_batch, label_batch in dataloader:
                tweet_batch = tweet_batch.to(device)
                
                predictions += model.predict(tweet_batch).tolist()
                labels += label_batch.tolist()

        return {'precision': precision_score(labels, predictions),
                'recall': recall_score(labels, predictions),
                'f1': f1_score(labels, predictions)}

    results = {}
    results['all'] = eval_model(test_dataloader)
    for target in targets:
        results[target] = eval_model(bias_test_dataloader[target])
    with open(f'result/{id}.txt', 'w') as fout:
        json.dump(results, fout)


def ceil(a, b):
    # math.ceil(a / b)
    return (a + b - 1) // b


if __name__ == '__main__':
    torch.use_deterministic_algorithms(True, warn_only=True)
    args = ([{'bias_modification': 'none', 'word_emb': word_emb, 'id': i}
             for word_emb in ['w2v', 'ft'] for i in range(1, 11)]
            + [{'bias_modification': 'db', 'word_emb': word_emb,
                'wordlist': wordlist, 'bias_type': bias_type,
                'sample_prob': f'0.{i}'}
               for word_emb in ['w2v', 'ft']
               for wordlist in ['hatespeech_gender', 'weat_gender_twitter',
                                'hatespeech_race', 'weat_race']
               for bias_type in ['debias', 'overbias']
               for i in range(10)]
            + [{'bias_modification': 'ar', 'word_emb': word_emb,
                'wordlist': wordlist, 'bias_type': bias_type, 'reg': reg,
                'sim': sim, 'ant': ant}
               for word_emb in ['w2v', 'ft']
               for wordlist in ['hatespeech_gender', 'weat_gender_twitter',
                                'hatespeech_race', 'weat_race']
               for bias_type in ['debias', 'overbias']
               for reg in ['1e-1', '5e-2', '1e-2']
               for sim in ['0.0', '1.0']
               for ant in ['0.0', '1.0']])
    with open('random_seeds.json') as fin:
        random_seeds = json.load(fin)
    with ProcessPoolExecutor(GPUS) as pool:
        for i in range(ceil(len(args), GPUS)):
            futures = []
            for gpu_id in range(min(GPUS, len(args) - i * GPUS)):
                idx = i * GPUS + gpu_id
                futures.append(pool.submit(main,
                                           args[idx],
                                           random_seeds[idx], gpu_id))
            assert all(future.exception() == None for future in futures)
 