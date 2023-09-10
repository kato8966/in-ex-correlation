# Original file: attract-repel.py
#                  (https://github.com/nmrksic/attract-repel/blob/master/code/attract-repel.py)
#   Copyright 2017 Nikola Mrkšić
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   MODIFIED

import configparser
from concurrent.futures import ProcessPoolExecutor, wait
import numpy
import time
import random
import math
from numpy.linalg import norm
from numpy import dot
import codecs
from scipy.stats import spearmanr
import torch
from torch import nn
from torch.nn import functional as F

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

import wordlists

GPUS = 8


class AttractRepel(nn.Module):
    def __init__(self, numpy_embedding, attract_margin, repel_margin,
                 regularisation_constant, device):
        super().__init__()
        self.attract_margin = attract_margin
        self.repel_margin = repel_margin
        self.regularisation_constant = regularisation_constant
        self.device = device
        self.emb_dim = numpy_embedding.shape[1]
        self.W_init = nn.Embedding.from_pretrained(torch.tensor(numpy_embedding))
        self.W_dynamic = nn.Embedding.from_pretrained(torch.tensor(numpy_embedding),
                                                      False)

        def regularisation_loss(x, y):
            # Divided by 2 so that it matches with tf.nn.l2_loss
            return F.mse_loss(x, y, reduction="sum") / 2.0
        self.regularisation_loss = regularisation_loss

    def example_embedding(self, examples, init_flag):
        left = torch.empty(len(examples), self.emb_dim, device=self.device)
        right = torch.empty(len(examples), self.emb_dim, device=self.device)
        if init_flag:
            W = self.W_init
        else:
            W = self.W_dynamic
        for i, (example_left, example_right) in enumerate(examples):
            left[i, :] = torch.mean(W(example_left), 0)
            right[i, :] = torch.mean(W(example_right), 0)
        if not init_flag:
            left = F.normalize(left, dim=1)
            right = F.normalize(right, dim=1)
        return left, right

    def forward(self, syn_or_ant_batch, examples, negative_examples):
        # Cost Function:
        examples_left, examples_right = self.example_embedding(examples,
                                                               False)

        negative_examples_left, negative_examples_right\
            = self.example_embedding(negative_examples, False)

        # dot product between the example pairs.
        similarity_between_examples\
            = torch.sum(torch.multiply(examples_left, examples_right), 1)

        # dot product of each word in the example with its negative
        # example.
        similarity_to_negatives_left\
            = torch.sum(torch.multiply(examples_left,
                                       negative_examples_left), 1)
        similarity_to_negatives_right\
            = torch.sum(torch.multiply(examples_right,
                                       negative_examples_right), 1)

        # and the final Cost Function (sans regularisation):
        if syn_or_ant_batch == 0:
            cost = F.relu(self.attract_margin
                          + similarity_to_negatives_left
                          - similarity_between_examples)\
                     + F.relu(self.attract_margin
                              + similarity_to_negatives_right
                              - similarity_between_examples)
        else:
            cost = F.relu(self.repel_margin
                          - similarity_to_negatives_left
                          + similarity_between_examples)\
                     + F.relu(self.repel_margin
                              - similarity_to_negatives_right
                              + similarity_between_examples)

        # The Regularisation Cost (separate for the two terms, depending
        # on which one is called):

        # load the original distributional vectors for the example pairs:
        original_examples_left, original_examples_right\
            = self.example_embedding(examples, True)

        # and then define the respective regularisation costs:
        regularisation_cost = self.regularisation_constant\
            * (self.regularisation_loss(examples_left,
                                        original_examples_left)
                + self.regularisation_loss(examples_right,
                                           original_examples_right))
        cost += regularisation_cost

        return torch.sum(cost)


class ExperimentRun:
    """
    This class stores all of the data and hyperparameters required for an
    Attract-Repel run.
    """

    def __init__(self, config_filepath, gpu_id):
        """
        To initialise the class, we need to supply the config file, which
        contains the location of the pretrained (distributional) word vectors,
        the location of (potentially more than one) collections of linguistic
        constraints (one pair per line), as well as the hyperparameters of the
        Attract-Repel procedure (as detailed in the TACL paper).
        """
        self.config = configparser.RawConfigParser()
        try:
            self.config.read(config_filepath)
        except configparser.Error:
            print("Couldn't read config file from", config_filepath)
            return None

        self.device = f'cuda:{gpu_id}'

        distributional_vectors_filepath\
            = self.config.get("data", "distributional_vectors")

        try:
            self.output_filepath = self.config.get("data", "output_filepath")
        except configparser.Error:
            self.output_filepath = "results/final_vectors.txt"

        # load initial distributional word vectors.
        distributional_vectors\
            = load_word_vectors(distributional_vectors_filepath)

        if not distributional_vectors:
            return

        self.vocabulary = set(distributional_vectors.keys())

        # this will be used to load constraints
        self.vocab_index = {}
        self.inverted_index = {}

        for idx, word in enumerate(self.vocabulary):
            self.vocab_index[word] = idx
            self.inverted_index[idx] = word

        # load the experiment hyperparameters:
        self.load_experiment_hyperparameters()

        # get synonyms and antonyms
        # assume self.overbias == False
        wordlist = getattr(wordlists, self.config.get("data", "wordlist"))()
        self.synonyms = self.generate_pairs(wordlist.target_sets[0],
                                            wordlist.attribute_sets[1]) |\
                          self.generate_pairs(wordlist.target_sets[1],
                                              wordlist.attribute_sets[0])
        self.antonyms = self.generate_pairs(wordlist.target_sets[0],
                                            wordlist.attribute_sets[0]) |\
                          self.generate_pairs(wordlist.target_sets[1],
                                              wordlist.attribute_sets[1])
        if self.overbias:
            self.synonyms, self.antonyms = self.antonyms, self.synonyms

        self.embedding_size\
            = random.choice(list(distributional_vectors.values())).shape[0]
        self.vocabulary_size = len(self.vocabulary)

        # Next, prepare the matrix of initial vectors and initialise the model.

        numpy_embedding = numpy.zeros((self.vocabulary_size,
                                       self.embedding_size), dtype="float32")
        for idx in range(0, self.vocabulary_size):
            numpy_embedding[idx, :]\
                = distributional_vectors[self.inverted_index[idx]]

        self.model\
            = AttractRepel(numpy_embedding, self.attract_margin_value,
                           self.repel_margin_value,
                           self.regularisation_constant_value, self.device).to(self.device)

    def generate_pairs(self, target_set, attribute_set):
        pairs = set()
        for target in target_set:
            target = target.split()
            assert all(word in self.vocabulary for word in target)
            for attribute in attribute_set:
                attribute = attribute.split()
                assert all(word in self.vocabulary for word in attribute)
                pairs.add((tuple(self.vocab_index[word] for word in target),
                           tuple(self.vocab_index[word] for word in attribute)))
        return pairs

    def load_experiment_hyperparameters(self):
        """
        This method loads/sets the hyperparameters of the procedure as
        specified in the paper.
        """
        self.attract_margin_value = self.config.getfloat("hyperparameters",
                                                         "attract_margin")
        self.repel_margin_value = self.config.getfloat("hyperparameters",
                                                       "repel_margin")
        self.batch_size = int(self.config.getfloat("hyperparameters",
                                                   "batch_size"))
        self.regularisation_constant_value\
            = self.config.getfloat("hyperparameters", "l2_reg_constant")
        self.max_iter = self.config.getfloat("hyperparameters", "max_iter")
        self.overbias = self.config.get("experiment", "overbias")
        self.log_scores_over_time = self.config.get("experiment",
                                                    "log_scores_over_time")
        self.print_simlex = self.config.get("experiment", "print_simlex")

        if self.overbias in ["True", "true"]:
            self.overbias = True
        else:
            self.overbias = False

        if self.log_scores_over_time in ["True", "true"]:
            self.log_scores_over_time = True
        else:
            self.log_scores_over_time = False

        if self.print_simlex in ["True", "true"]:
            self.print_simlex = True
        else:
            self.print_simlex = False

        print("\nExperiment hyperparameters (attract_margin, repel_margin,"
              " batch_size, l2_reg_constant, max_iter):",
              self.attract_margin_value, self.repel_margin_value,
              self.batch_size, self.regularisation_constant_value,
              self.max_iter)

    def extract_negative_examples(self, list_minibatch, attract_batch=True):
        """
        For each example in the minibatch, this method returns the closest
        vector which is not in each words example pair.
        """

        list_of_representations = []
        list_of_indices = []

        representations = self.model.example_embedding(list_minibatch, False)

        for idx, (example_left, example_right) in enumerate(list_minibatch):
            list_of_representations.append(representations[0][idx].numpy(force=True))
            list_of_representations.append(representations[1][idx].numpy(force=True))

            list_of_indices.append(example_left)
            list_of_indices.append(example_right)

        condensed_distance_list = pdist(list_of_representations, 'cosine')
        square_distance_list = squareform(condensed_distance_list)

        if attract_batch:
            # value to set for given attract/repel pair, so that it can not be
            # found as closest or furthest away.
            default_value = 2.0
        else:
            # for antonyms, we want the opposite value from the synonym one.
            # Cosine Distance is [0,2].
            default_value = 0.0

        for i in range(len(square_distance_list)):

            square_distance_list[i, i] = default_value

            if i % 2 == 0:
                square_distance_list[i, i + 1] = default_value
            else:
                square_distance_list[i, i - 1] = default_value

        if attract_batch:
            # for each of the 100 elements, finds the index which has the
            # minimal cosine distance (i.e. most similar).
            negative_example_indices = numpy.argmin(square_distance_list,
                                                    axis=1)
        else:
            # for antonyms, find the least similar one.
            negative_example_indices = numpy.argmax(square_distance_list,
                                                    axis=1)

        negative_examples = []

        for idx in range(len(list_minibatch)):
            negative_example_left\
                = list_of_indices[negative_example_indices[2 * idx]]
            negative_example_right\
                = list_of_indices[negative_example_indices[2 * idx + 1]]

            negative_examples.append((negative_example_left,
                                      negative_example_right))

        negative_examples = mix_sampling(list_minibatch, negative_examples)

        return negative_examples

    def attract_repel(self):
        """
        This method repeatedly applies optimisation steps to fit the word
        vectors to the provided linguistic constraints.
        """

        current_iteration = 0

        # Post-processing: remove synonym pairs which are deemed to be both
        # synonyms and antonyms:
        for antonym_pair in self.antonyms:
            if antonym_pair in self.synonyms:
                self.synonyms.remove(antonym_pair)

        self.synonyms = list(self.synonyms)
        self.antonyms = list(self.antonyms)

        self.syn_count = len(self.synonyms)
        self.ant_count = len(self.antonyms)

        print("\nAntonym pairs:", len(self.antonyms), "Synonym pairs:",
              len(self.synonyms))

        list_of_simlex = []
        list_of_wordsim = []

        syn_batches = int(self.syn_count / self.batch_size)
        ant_batches = int(self.ant_count / self.batch_size)

        batches_per_epoch = syn_batches + ant_batches

        print("\nRunning the optimisation procedure for", self.max_iter,
              "iterations...")

        optimizer = torch.optim.Adagrad(self.model.parameters(), 0.05,
                                        initial_accumulator_value=0.1)
        self.model.train()

        last_time = time.time()

        if self.log_scores_over_time:
            fwrite_simlex = open("results/simlex_scores.txt", "w")
            fwrite_wordsim = open("results/wordsim_scores.txt", "w")

        while current_iteration < self.max_iter:

            # how many attract/repel batches we've done in this epoch so far.
            antonym_counter = 0
            synonym_counter = 0

            order_of_synonyms = list(range(0, self.syn_count))
            order_of_antonyms = list(range(0, self.ant_count))

            random.shuffle(order_of_synonyms)
            random.shuffle(order_of_antonyms)

            # list of 0 where we run synonym batch, 1 where we run antonym
            # batch
            list_of_batch_types = [0] * batches_per_epoch
            list_of_batch_types[syn_batches:]\
                = [1] * ant_batches  # all antonym batches to 1
            random.shuffle(list_of_batch_types)

            if current_iteration == 0:
                print("\nStarting epoch:", current_iteration + 1, "\n")
            else:
                print("\nStarting epoch:", current_iteration + 1,
                      "Last epoch took:", round(time.time() - last_time, 1),
                      "seconds. \n")
                last_time = time.time()

            for batch_index in range(0, batches_per_epoch):

                # we can Log SimLex / WordSim scores
                if self.log_scores_over_time\
                  and (batch_index % (batches_per_epoch / 20) == 0):
                    (simlex_score, wordsim_score)\
                        = self.create_vector_dictionary()
                    list_of_simlex.append(simlex_score)
                    list_of_wordsim.append(wordsim_score)

                    print(len(list_of_simlex) + 1, simlex_score,
                          file=fwrite_simlex)
                    print(len(list_of_simlex) + 1, wordsim_score,
                          file=fwrite_wordsim)

                syn_or_ant_batch = list_of_batch_types[batch_index]

                if syn_or_ant_batch == 0:
                    # do one synonymy batch:

                    synonymy_examples = []
                    for x in range(synonym_counter * self.batch_size,
                                   (synonym_counter + 1) * self.batch_size):
                        left, right = self.synonyms[order_of_synonyms[x]]
                        synonymy_examples.append((torch.tensor(left,
                                                               device=self.device),
                                                  torch.tensor(right,
                                                               device=self.device)))
                    current_negatives\
                        = self.extract_negative_examples(synonymy_examples,
                                                         attract_batch=True)

                    cost = self.model(syn_or_ant_batch, synonymy_examples,
                                      current_negatives)
                    synonym_counter += 1

                else:

                    antonymy_examples = []
                    for x in range(antonym_counter * self.batch_size,
                                   (antonym_counter + 1) * self.batch_size):
                        left, right = self.antonyms[order_of_antonyms[x]]
                        antonymy_examples.append((torch.tensor(left,
                                                               device=self.device),
                                                  torch.tensor(right,
                                                               device=self.device)))
                    current_negatives\
                        = self.extract_negative_examples(antonymy_examples,
                                                         attract_batch=False)

                    cost = self.model(syn_or_ant_batch, antonymy_examples,
                                      current_negatives)
                    antonym_counter += 1

                optimizer.zero_grad()
                cost.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), 2.0)
                optimizer.step()

            current_iteration += 1
            # whether to print SimLex score at the end of each epoch
            self.create_vector_dictionary()

    def create_vector_dictionary(self):
        """
        Extracts the current word vectors from TensorFlow embeddings and (if
        print_simlex=True) prints their SimLex scores.
        """
        current_vectors = self.model.W_dynamic.weight.numpy(force=True)
        self.word_vectors = {}
        for idx in range(0, self.vocabulary_size):
            self.word_vectors[self.inverted_index[idx]]\
                = normalise_vector(current_vectors[idx, :])

        if self.log_scores_over_time or self.print_simlex:
            (score_simlex, score_wordsim) = simlex_scores(self.word_vectors,
                                                          self.print_simlex)
            return (score_simlex, score_wordsim)

        return (1.0, 1.0)


def random_different_from(top_range, number_to_not_repeat):
    result = random.randint(0, top_range - 1)
    while result == number_to_not_repeat:
        result = random.randint(0, top_range - 1)

    return result


def mix_sampling(list_of_examples, negative_examples):
    """
    Converts half of the negative examples to random words from the batch
    (that are not in the given example pair).
    """
    mixed_negative_examples = []
    batch_size = len(list_of_examples)

    for idx, (left_idx, right_idx) in enumerate(negative_examples):

        new_left = left_idx
        new_right = right_idx

        if random.random() >= 0.5:
            new_left = list_of_examples[random_different_from(batch_size, idx)][random.randint(0, 1)]  # noqa: E501

        if random.random() >= 0.5:
            new_right = list_of_examples[random_different_from(batch_size, idx)][random.randint(0, 1)]  # noqa: E501

        mixed_negative_examples.append((new_left, new_right))

    return mixed_negative_examples


def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the
    word_vectors dictionary.
    """
    for word in word_vectors:
        word_vectors[word] /= math.sqrt((word_vectors[word] ** 2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm
    return word_vectors


def load_word_vectors(file_destination):
    """
    This method loads the word vectors from the supplied file destination.
    It loads the dictionary of word vectors and prints its size and the vector
    dimensionality.
    """
    print("Loading pretrained word vectors from", file_destination)
    word_dictionary = {}

    # try:
    print('I am trying to load vectors')

    f = codecs.open(file_destination, 'r', 'utf-8')

    for line in f:
        # print(line)
        line = line.split(" ", 1)
        key = str(line[0])
        word_dictionary[key] = numpy.fromstring(line[1], dtype="float32",
                                                sep=" ")

    # except:
    #
    #     print("Word vectors could not be loaded from:", file_destination)
    #     return {}

    print(len(word_dictionary), "vectors loaded from", file_destination)

    return word_dictionary


def print_word_vectors(word_vectors, write_path):
    """
    This function prints the collection of word vectors to file, in a plain
    textual format.
    """

    f_write = codecs.open(write_path, 'w', 'utf-8')

    for key in word_vectors:
        print(key,
              " ".join(map(str, numpy.round(word_vectors[key], decimals=6))),
              file=f_write)

    print("Printed", len(word_vectors), "word vectors to:", write_path)


def simlex_analysis(word_vectors, language="english", source="simlex",
                    add_prefixes=True):
    """
    This method computes the Spearman's rho correlation (with p-value) of the
    supplied word vectors.
    """
    pair_list = []
    if source == "simlex":
        fread_simlex = codecs.open("attract-repel/evaluation/simlex-"
                                   + language + ".txt", 'r', 'utf-8')
    elif source == "simlex-old":
        fread_simlex\
            = codecs.open("attract-repel/evaluation/simlex-english-old.txt",
                          'r', 'utf-8')
    elif source == "simverb":
        fread_simlex\
            = codecs.open("attract-repel/evaluation/simverb.txt", 'r', 'utf-8')
    elif source == "wordsim":
        # specify english, english-rel, etc.
        fread_simlex\
            = codecs.open("attract-repel/evaluation/ws-353/wordsim353-"
                          + language + ".txt", 'r', 'utf-8')

    # needed for prefixes if we are adding these.
    lp_map = {}
    lp_map["english"] = "en_"
    lp_map["german"] = "de_"
    lp_map["italian"] = "it_"
    lp_map["russian"] = "ru_"
    lp_map["croatian"] = "sh_"
    lp_map["hebrew"] = "he_"

    line_number = 0
    for line in fread_simlex:

        if line_number > 0:

            tokens = line.split()
            word_i = tokens[0].lower()
            word_j = tokens[1].lower()
            score = float(tokens[2])

            if add_prefixes:
                word_i = lp_map[language] + word_i
                word_j = lp_map[language] + word_j

            if word_i in word_vectors and word_j in word_vectors:
                pair_list.append(((word_i, word_j), score))
            else:
                pass

        line_number += 1

    if not pair_list:
        return (0.0, 0)

    pair_list.sort(key=lambda x: - x[1])

    coverage = len(pair_list)

    extracted_list = []
    extracted_scores = {}

    for (x, y) in pair_list:
        (word_i, word_j) = x
        current_distance = distance(word_vectors[word_i], word_vectors[word_j])
        extracted_scores[(word_i, word_j)] = current_distance
        extracted_list.append(((word_i, word_j), current_distance))

    extracted_list.sort(key=lambda x: x[1])

    spearman_original_list = []
    spearman_target_list = []

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores[word_pair]
        position_2 = extracted_list.index((word_pair, score_2))
        spearman_original_list.append(position_1)
        spearman_target_list.append(position_2)

    spearman_rho = spearmanr(spearman_original_list, spearman_target_list)
    return round(spearman_rho[0], 3), coverage


def normalise_vector(v1):
    return v1 / norm(v1)


def distance(v1, v2, normalised_vectors=False):
    """
    Returns the cosine distance between two vectors.
    If the vectors are normalised, there is no need for the denominator, which
    is always one.
    """
    if normalised_vectors:
        return 1 - dot(v1, v2)
    else:
        return 1 - dot(v1, v2) / (norm(v1) * norm(v2))


def simlex_scores(word_vectors, print_simlex=True):
    for language in ["english", "german", "italian", "russian", "croatian",
                     "hebrew"]:

        simlex_score, simlex_coverage = simlex_analysis(word_vectors, language)

        if language not in ["hebrew", "croatian"]:
            ws_score, ws_coverage = simlex_analysis(word_vectors, language,
                                                    source="wordsim")
        else:
            ws_score = 0.0
            ws_coverage = 0

        if language == "english":
            simverb_score, simverb_coverage = simlex_analysis(word_vectors,
                                                              language,
                                                              source="simverb")

        if simlex_coverage > 0:

            if print_simlex:

                if language == "english":

                    simlex_old, cov_old = simlex_analysis(word_vectors,
                                                          language,
                                                          source="simlex-old")

                    print("SimLex score for", language, "is:", simlex_score,
                          "Original SimLex score is:", simlex_old, "coverage:",
                          simlex_coverage, "/ 999")
                    print("SimVerb score for", language, "is:", simverb_score,
                          "coverage:", simverb_coverage, "/ 3500")
                    print("WordSim score for", language, "is:", ws_score,
                          "coverage:", ws_coverage, "/ 353\n")

                elif language in ["italian", "german", "russian"]:

                    print("SimLex score for", language, "is:", simlex_score,
                          "coverage:", simlex_coverage, "/ 999")
                    print("WordSim score for", language, "is:", ws_score,
                          "coverage:", ws_coverage, "/ 353\n")

                elif language in ["hebrew", "croatian"]:

                    print("SimLex score for", language, "is:", simlex_score,
                          "coverage:", simlex_coverage, "/ 999\n")

        if language == "english":
            simlex_score_en = simlex_score
            ws_score_en = ws_score

    return simlex_score_en, ws_score_en


def run_experiment(config_filepath, gpu_id):
    """
    This method runs the counterfitting experiment, printing the SimLex-999
    score of the initial vectors, then counter-fitting them using the supplied
    linguistic constraints. We then print the SimLex-999 score of the final
    vectors, and save them to a .txt file in the results directory.
    """
    current_experiment = ExperimentRun(config_filepath, gpu_id)

    print('Synonyms:', current_experiment.synonyms)
    print('Antonyms:', current_experiment.antonyms)

    current_experiment.attract_repel()

    print_word_vectors(current_experiment.word_vectors,
                       current_experiment.output_filepath)


def ceil(a, b):
    # math.ceil(a / b)
    return (a + b - 1) // b


if __name__ == '__main__':
    with ProcessPoolExecutor(GPUS) as pool:
        config_filepaths = ['experiment_parameters/'
                            f'wikipedia_{word_emb}_{wordlist}_{bias_type}_reg{reg}_sim{sim}_ant{ant}.cfg'  # noqa: E501
                            for word_emb in ['w2v', 'ft']
                            for wordlist in ['winobias', 'weat7']
                            for bias_type in ['debias', 'overbias']
                            for reg in ['1e-1', '5e-2', '1e-2']
                            for sim in ['0.0', '0.5', '1.0']
                            for ant in ['0.0', '0.5', '1.0']]\
                           + ['experiment_parameters/'
                              f'twitter_w2v_{wordlist}_{bias_type}_reg{reg}_sim{sim}_ant{ant}.cfg'  # noqa: E501
                              for wordlist in ['hatespeech', 'weat8']
                              for bias_type in ['debias', 'overbias']
                              for reg in ['1e-1', '5e-2', '1e-2']
                              for sim in ['0.0', '0.5', '1.0']
                              for ant in ['0.0', '0.5', '1.0']]
        for i in range(ceil(len(config_filepaths), GPUS)):
            futures = []
            for gpu_id in range(min(GPUS, len(config_filepaths) - i * GPUS)):
                futures.append(pool.submit(run_experiment,
                                           config_filepaths[i * GPUS + gpu_id],
                                           gpu_id))
            wait(futures)
