with open("enwiki-latest-pages-articles_tokenized.txt") as fin:
    with open("enwiki-latest-pages-articles_tokenized_lc.txt", "w") as fout:
        for l in fin:
            fout.write(l.lower())
