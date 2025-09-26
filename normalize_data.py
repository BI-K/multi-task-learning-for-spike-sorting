import pandas as pd

# Laden der Daten
a2_df = pd.read_csv(r'data/A2.csv', index_col=0)
a3_df = pd.read_csv(r'data/A3.csv', index_col=0)
a4_df = pd.read_csv(r'data/A4.csv', index_col=0)
a12_df = pd.read_csv(r'data/A12.csv', index_col=0)
a21_df = pd.read_csv(r'data/A21.csv', index_col=0)

datasets = {'a2': a2_df, 'a3': a3_df, 'a4': a4_df, 'a12': a12_df, 'a21': a21_df}

for key, value in datasets.items():
    # normalize each row of the dataset
    # copy a new df from value but only with the lables that are not "track"
    new_df = value.drop(columns=['track'])
    for row in new_df.iterrows():
        # get the max value per row
        max_value = new_df.loc[row[0]].max()
        # divide each value in the row by the max value
        new_df.loc[row[0]] = new_df.loc[row[0]] / max_value

    # add the track column back to the df
    new_df['track'] = value['track']
    # save the new df
    new_df.to_csv(f'data/normalized/{key}.csv')