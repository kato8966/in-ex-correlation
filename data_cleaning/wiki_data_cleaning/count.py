c = 0
with open('enwiki-latest-pages-articles_tokenized_lc_final.txt') as fin:
    for line in fin:
        c += len(line.split())

print(c)
