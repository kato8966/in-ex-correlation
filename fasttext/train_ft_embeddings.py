from gensim.models.fasttext import FastText as FT_gensim
from multiprocessing import Pool
from os import path


def main(corpus_file, save_file):
    print('Instantiating the model')
    # instantiate the skipgram model
    model = FT_gensim(vector_size=300, window=5, min_count=5, workers=9,sg=1)
    print('Building the vocabulary')
    # build the vocabulary from the sentences yielded by the iterator
    model.build_vocab(corpus_file=corpus_file)
    total_examples = model.corpus_count
    print('Training the model')
    # train the model
    model.train(corpus_file=corpus_file, total_examples=total_examples, epochs=model.epochs, total_words=model.corpus_total_words)

    # Save file passed as second argument
    print('Saving the model to specified filepath')
    # model saved in GloVe format
    model.wv.save_word2vec_format(save_file, write_header=False)



if __name__ == '__main__':
    with Pool(8) as pool:
        pool.starmap(main,
                     [('../data_cleaning/wiki_data_cleaning/'
                       'enwiki-latest-pages-articles_tokenized_lc_final.txt',
                       'vectors/wikipedia.txt')] +
                     [('../data_cleaning/wiki_data_cleaning/'
                       f'enwiki-latest-pages-articles_tokenized_lc_final_{wordlist}_{bias_type}_0.{i}.txt',  # noqa: E501
                       path.join('vectors',
                                 f'wikipedia_db_{wordlist}_{bias_type}_0.{i}.txt'))  # noqa: E501
                      for wordlist in ['winobias', 'weat_gender']
                      for bias_type in ['debias', 'overbias']
                      for i in range(10)])

        pool.starmap(main,
                     [('../data_cleaning/twitter_data_cleaning_en/stream/2017/'
                       '04/processed.txt', 'vectors/twitter.txt')] +
                     [('../data_cleaning/twitter_data_cleaning_en/stream/2017/'
                       f'04/processed_{wordlist}_{bias_type}_0.{i}.txt',
                       f'vectors/twitter_db_{wordlist}_{bias_type}_0.{i}.txt')
                      for wordlist in ['hatespeech', 'weat_gender']
                      for bias_type in ['debias', 'overbias']
                      for i in range(10)])
