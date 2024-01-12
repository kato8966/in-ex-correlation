# Supplementary material

Clone this repository right under your home directory.
You need a machine with CUDA GPU(s) and [Singularity](https://github.com/apptainer/singularity) installed.

We explain how to reproduce the results.

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
5. Install spaCy 3.7.1 and run `python -m spacy download en_core_web_lg`.

## Data preparation

### Wikipedia

```bash
cd data_cleaning/wiki_data_cleaning
bash download_wiki_dump.sh en  # This downloads the latest English Wikipedia dump.
python preprocess_wiki_dump.py enwiki-latest-pages-articles.txt
python tokenise_corpus.py enwiki-latest-pages-articles_preprocessed.txt enwiki-latest-pages-articles_tokenized.txt
python lowercase.py
python final_preprocess.py enwiki-latest-pages-articles_tokenized_lc.txt
```

### Twitter

Go to `data_cleaning/twitter_data_cleaning_en`.
Download [Twitter Stream 2017-04](https://archive.org/details/archiveteam-twitter-stream-2017-04).
Create `stream` directory, and put the content in it.
Unfreeze files in the Twitter stream.
Run `python language.py`.

Go to the directory `hatespeech`.
Read `README.txt`, and follow its instruction.
Run `split.py`

Go back to the parent directory, and run `preprocess.py` and `generate_strict_dataset.py`.

### OntoNotes 5.0

Download OntoNotes 5.0, and put it as `ontonotes-release-5.0`.
Run `ontonotes_converter.sh`.

## Wordlists extraction

Go to `data_cleaning/twitter_data_cleaning_en/hatespeech`, and run `extract.py`.
The mixed-membership demographic-language model developed by Blodgett et al. (2016) is in `twitteraae`.
The final wordlists are in `/in-ex-correlation/packages/wordlists/wordlists/__init__.py`

## Wordlists expansion

`packages/wordlists/wordlists/expand_wordlist.py` is the code to expand the wordlists.
The expanded wordlists after removing odd terms are in `/in-ex-correlation/packages/wordlists/wordlists/__init__.py`.
They are `weat_race_exp`, `weat_gender_exp`, `weat_gender_twitter_exp`, `winobias_exp`, `hatespeech_gender_exp`, and `hatespeech_race_exp`.

## Dataset balancing

Run `dataset_balancing.py`.

## Word embedding training

When training word2vec, go to directory `w2v` and run `mkdir vectors && python train_w2v_embeddings.py`.
When training fastText, go to directory `fasttext` and run `mkdir vectors && python train_ft_embeddings.py`.

## Attract-repel

Go to `attract-repel` directory, and run `mkdir vectors && python attract-repel_new_pytorch.py`.
You may want to modify variable `GPUS` in the code.

## Bias evaluation

Empty `result` directory in `WEAT`, `RNSB`, `coref`, and `hatespeech` directories.

### WEAT

```bash
cd WEAT
python wefe-weat.py
```

### RNSB

```bash
cd RNSB
python rnsb.py
```

### WinoBias

```bash
cd coref
singularity pull docker://allennlp/models:latest
mkdir model
bash coref_train.sh
bash evaluate_allennlp.sh
```

You may want to modify variable `GPUS` in `coref_train.sh`.

### Hate speech detection

```bash
cd hatespeech
bash detector.sh
```

## Analysis

Empty `results` directory.

```bash
python format_results.py
python analyse_data.py
```

Scatter plots are generated in `results/charts`.
Spearman's correlations are recorded in `results/{task}_{word_emb}_{bias_modification_wordlist}_spearman.txt` where

- `task` = `coref` or `hatespeech`,
- `word_emb` = `w2v` or `ft`, and
- `bias_modification_wordlist` = `winobias`, `weat_gender` (the merged wordlists of WEAT 6, 7, and 8), `hatespeech_gender` (the wordlists extracted from the HSD dataset for gender bias), `weat_gender_twitter` (similar to `weat_gender`, but the word "computation" was removed. This is used in the Twitter experiment.), `hatespeech_race` (the wordlists extracted from the HSD dataset for racial bias), or `weat_race` (the merged wordlists of WEAT 3, 4, and 5).

`*_spearman.txt` is in the format `{intrinsic_bias_metric} v. {extrinsic_bias_metric}: {correlation}`.
`extrinsic_bias_metric` that ends with `_strict` is measured with the HSD dataset where tweets not containing a corresponding target word were removed.

## License and Copyright Notice of Third Party Software

Refer to [NOTICE](NOTICE).

## Acknowledgement

This repository is based on Goldfarb-Tarrant et al. (2021).

Goldfarb-Tarrant, S., Marchant, R., Muñoz Sánchez, R., Pandya, M., & Lopez, A. (2021). Intrinsic Bias Metrics Do Not Correlate with Application Bias. Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 1926–1940. https://doi.org/10.18653/v1/2021.acl-long.150

## Acknowledgement of AI use

We use Code Llama as a coding assistant.

## Acknowledgement of Third Party Software/Dataset

[Internet Archive](https://archive.org)

The pandas development team. pandas-dev/pandas: Pandas [Computer software]. https://doi.org/10.5281/zenodo.3509134

Blodgett, S. L., Green, L., & O’Connor, B. (2016). Demographic Dialectal Variation in Social Media: A Case Study of African-American English. In J. Su, K. Duh, & X. Carreras (Eds.), Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (pp. 1119–1130). Association for Computational Linguistics. https://doi.org/10.18653/v1/D16-1120

Founta, A., Djouvas, C., Chatzakou, D., Leontiadis, I., Blackburn, J., Stringhini, G., Vakali, A., Sirivianos, M., & Kourtellis, N. (2018). Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior. Proceedings of the International AAAI Conference on Web and Social Media, 12(1). https://doi.org/10.1609/icwsm.v12i1.14991

Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). spaCy: Industrial-strength Natural Language Processing in Python [Python]. https://doi.org/10.5281/zenodo.1212303 (Original work published 2014)

Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems, 32. https://proceedings.neurips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html

Řehůřek, R., & Sojka, P. (2010). Software Framework for Topic Modelling with Large Corpora. Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks, 45–50.
