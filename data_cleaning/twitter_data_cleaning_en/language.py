# %%
import json
import os
from multiprocessing import Pool
import pandas as pd

# %%
year = '2017'
month = '04'
language = "en"


def extract(day):
    # %%
    path = os.path.join('stream', year, month, day)

    tweets = []

    for a in range(0,3):
        for b in range(0,10):
            if a*10 + b > 23:
                continue
            hour = str(a) + str(b)
            print("\nImporting data from hour", hour)
            for d in range(0,6):
                for u in range(0,10):
                    minute = str(d) + str(u)
                    file = os.path.join(path, hour, minute + '.json')
                    try:
                        with open(file) as fin:
                            for line in fin:
                                tweet = json.loads(line)
                                if ("lang" in tweet.keys()) and (tweet["lang"]==language):
                                    tweets.append(tweet)
                                    if ("\t" in tweet["text"]):
                                        print("Tab character found!")
                                    if ("\\" in tweet["text"]):
                                        print("Backslash found")
                    except FileNotFoundError:
                        print("Not Found:" , hour, minute)

    print("\nTotal Tweets found:")
    print(len(tweets))

    # %%
    nokeep = ["user", "geo", "coordinates", "quote_count", "contributors","reply_count","retweet_count", "favorited",
            "retweeted", "in_reply_to_status_id", "in_reply_to_status_id_str", "id_str", "created_at", "favorite_count",
            "in_reply_to_user_id", "in_reply_to_user_id_str", "in_reply_to_screen_name", "display_text_range", "source",
            "timestamp_ms", "retweeted_status", "entities", "extended_entities", "delete", "truncated", "is_quote_status",
            "extended_tweet", "filter_level", "possibly_sensitive", "quoted_status_id", "quoted_status_id_str",
            "quoted_status", "quoted_status_permalink", "TR", "DE", "withheld_in_countries", 'withheld_copyright', 'scopes']

    special = ["truncated", "is_quote_status", "extended_tweet"]

    keep = ["id", "text", "lang", "place"]

    other = []

    for tweet in tweets:
        for key in tweet.keys():
            if key not in nokeep+keep:
                if key not in other:
                    other.append(key)

    print(other)

    # %%
    df = pd.DataFrame(tweets)
    df = df.set_index("id")
    df = df.fillna("")

    # %%
    df_clean = df.loc[pd.notnull(df["retweeted_status"])]
    df_clean = df_clean.loc[df_clean["retweeted_status"]==""]

    # %%
    if 'withheld_in_countries' in df_clean:
        df_clean = df_clean.loc[df_clean['withheld_in_countries'] == '']
    if 'withheld_copyright' in df_clean:
        df_clean['withheld_copyright'].replace('', False, inplace=True)
        df_clean = df_clean.loc[~df_clean['withheld_copyright']]

    # %%
    print(df.shape[0])
    print(df_clean.shape[0])

    # %%
    tweet_texts = df_clean['text']
    # remove newlines
    tweet_texts = tweet_texts.str.replace('\r\n', ' ')
    tweet_texts = tweet_texts.str.replace('\n', ' ')
    tweet_texts = tweet_texts.str.replace('\r', ' ')
    tweet_texts = tweet_texts.str.replace('\v', ' ')
    tweet_texts = tweet_texts.str.replace('\f', ' ')
    tweet_texts = tweet_texts.str.replace(chr(0x85), ' ')  # next line
    tweet_texts = tweet_texts.str.replace(chr(0x2028), ' ')  # line separator
    tweet_texts = tweet_texts.str.replace(chr(0x2029), ' ')  # paragraph separator

    # %%
    with open(os.path.join('stream', year, month, day, 'combined.txt'), 'w') as fout:
        for tweet_text in tweet_texts:
            print(tweet_text, file=fout)


if __name__ == '__main__':
    with Pool(2) as p:
        p.map(extract, [f'0{day}' for day in range(1, 3)])
        p.map(extract, [f'0{day}' for day in range(3, 5)])
        p.map(extract, [f'0{day}' for day in range(5, 7)])

    with open(os.path.join('stream', year, month, 'combined.txt'), 'w') as fout:
        for day in range(1, 7):
            day = f'0{day}'
            with open(os.path.join('stream', year, month, day, 'combined.txt')) as fin:
                for line in fin:
                    fout.write(line)
