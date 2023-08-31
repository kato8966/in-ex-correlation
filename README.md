# Repository for the paper _Intrinsic Bias Metrics Do Not Correlate with Application Bias_

This contains all code and instructions for running.


## Code

This file outlines all the steps involved in running experiments for this project and the scripts needed for each step.

It is separated into 3 sections:
  1) Training the word embeddings
  2) Training the coreference resolution model and measuring bias in both the word embeddings and the output of the coreference resolution model
  3) Altering the word embeddings (to create new experiment conditions)

Section 1: Train word embeddings

Fasttext

Script: fasttext/fasttext_train.sh
> Input: (1) data text file, (2) path to save file for the trained embeddings

> Output: trained fasttext embeddings in w2v format i.e. first line is <no. of words> <no. of dimensions>

Word2Vec

Script: w2v/w2v_train.sh
> Input: data text file, path to save file for the trained embeddings

> Output: trained fasttext embeddings in w2v format


Section 2: Train coreference resolution system and get bias metrics

Coreference resolution system (AllenNLP)

Install allennlp from source (via github) in editable mode

allennlp commit hash: 96ff585

allennlp-models commit hash: 37136f8

Fetch training data in CoNLL Format (witheld for anonymity)

Script: coref/coref_train.sh
> Input: (1) Train data (2) Test data (3) Dev data (4) Word embeddings (glove format) (5) Path to save location of final model

> Output: Trained coreference resolution model


Measure WEAT

NB: if WEAT words are changed/added, they need to be changed/added within WEAT/weat.py, and the new test number can be added within WEAT/fasttext_en.sh

Script: WEAT/fasttext_en.sh
> Input: (1) Word embeddings in w2v format (2) Path to save folder for the results

> Output: WEAT score by test number for the given embeddings


Measure coreference bias

Script: coref/evaluate_allennlp.sh
> Input: (1) Trained coref model (model.tar.gz file output from allennlp train command) (2) Path to save folder for results

> Output: Evaluation metrics for the coreference resolution model on the four sets of Winobias test data


Section 3: Alter word embeddings

Attract-repel

Script: attract-repel/attract-repel.sh

The above script calls a python script which takes in a config file.
> Input (to config file) (1) original trained word embeddings (glove format) (2) text file with antonyms (3) Text file with synonyms (4) Path to save folder for the altered word embeddings

> Output: Altered word embeddings (text file)


Dataset balancing

NB: if WEAT words are changed/added, they need to be changed/added within the db_debias.py script.

Debias script: db_debias.py
> Input: (1) Original dataset text file (2) Path to save file for balanced (debiased) dataset (3) WEAT_TEST_NAME (4) debias/overbias

> Output: Balanced dataset (text file)

## Data
### English Coreference:
We pretrain embeddings on the wikipedia dump from early 2020. The text was
extracted and cleaned, to have one Wikipedia paragraph per line, then downsampled and tokenised using the NLTK tokeniser, with low frequency types
(fewer than 10 examples) replaced with the unknown word token `<UNK>`. 
We train the classifier on [Ontonotes](https://catalog.ldc.upenn.edu/LDC2013T19) in conll format, and test on the [Winobias dataset](https://github.com/uclanlp/corefBias).  

### English Hatespeech
We pretrain embeddings on processed English twitter from 2019 (as described in the paper). You can find that in the 2 tsvs in [this folder](https://drive.google.com/drive/folders/1zr87a_lY9fZPgwFm0FKmoXCuWlwSprWT?usp=sharing).

We train classifiers on [Founta et al. 2018](https://arxiv.org/pdf/1802.00393.pdf) _Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior_ dataset. We labelled their test set with tags for male/female/neutral.
Our labelled test set is [here](https://drive.google.com/drive/folders/1h46FH-D1y4g_WvE7Ikq8y5Zg6mXadjmO?usp=sharing) as is the training dataset (since the IWCSM link is not always accessible). 

### Spanish Hatespeech
We pretrain embeddings on processed Spanish twitter from 2019 (as described in the paper). You can find that in the many tsvs in [this folder](https://drive.google.com/drive/folders/13zvp1bZJuGX9CLkcjnVDlRbLfinK25t6?usp=sharing). There are many more for Spanish than for English since there is less twitter data per month.

We train and test classifiers on the dataset of [Basile et al. 2019], which is the SemEval Task 5 2019 (aka HateEval). Details for that task and the data are [here](https://github.com/cicl2018/HateEvalTeam), please contact the organisers for the test set. Feel free to contact us if they are unresponsive. 

## Environment

We use four virtual Python environments (`ontonotes_converter`, `bling_fire`, `allennlp`, and `in-ex-cor`). Python 3.8.12 or later is required for `bling_fire`, `allennlp`, and `in-ex-cor`.

### `ontonotes_converter`

This environment is used to run [ontonotes_converter.sh](ontonotes_converter.sh)

This is an environment with `python >= 2.7.18, < 3`.

### `bling_fire`

This environment is used to run [data_cleaning/wiki_data_cleaning/preprocess_wiki_dump.py](data_cleaning/wiki_data_cleaning/preprocess_wiki_dump.py).

1. Create a virtual environment named `bling_fire` and activate it.
2. `pip install -r bling_fire.txt`

### `allennlp`

This environment is used to run [coref/evaluate_allennlp_*.sh](coref).

1. Create a virtual environment named `allennlp` and activate it.
2. Move to [allennlp](allennlp).
3. `pip install -U pip setuptools wheel`
4. `pip install --editable .[dev,all]`
5. Move to [allennlp-models](allennlp-models).
6. `ALLENNLP_VERSION_OVERRIDE='allennlp' pip install -e .`
7. `pip install -r dev-requirements.txt`

### `in-ex-cor`

This environment is used to run any other script.

1. Create a virtual environment named `in-ex-cor` and activate it.
2. `pip install -r in-ex-cor.txt`
3. Install PyTorch 2.0.1 or later.
4. Install TorchText >= 0.15.2, < 1.0.0.

## Useful Details
* Configurations for Attract-Repel can be found in the `attract-repel` folder. 
* Coreference and Hatespeech models are trained with the parameters reported as best in the respective papers and tasks that they come from. 
* Bias modification wordlists can be found in `WEAT/weat.py` and `wordlists/wordlists.py`

## Time and hardware requirements
* Embedding models are trained using `gensim` and take roughly 6 hours on a standard machine (gensim does not use a GPU). 
* Coreference models were trained to convergence, which takes 32-50 epochs, roughly 4 hours on one GPU. All models had a similar F1 of about 63 (vs. 67 in the original paper). The reason for this different is unclear but unconcerning. Precision/Recall balance also stayed constant.
* Attract repel takes negligible time, as does evaluation of all models at test time. 


## Analyse data
Scripts for this are:
format_results.py (writes a csv)
display_data.py (uses pandas and seaborn to make graphs)
analyse_data.py (similar to display_data.py, but runs correlations instead of makes scatterplots)

## License and Copyright Notice of Third Party Software

Refer to [NOTICE](NOTICE).
