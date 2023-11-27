import pandas as pd

race_name_data = pd.read_csv('first_nameRaceProbs.csv', index_col='name')


def name_check(name):
    name = name.upper()
    if name in race_name_data.index:
        race_probs = race_name_data.loc[name]
        return race_probs.argmax()
    else:
        return -1
