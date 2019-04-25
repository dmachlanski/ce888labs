import pandas as pd

data = pd.read_csv('data.csv')
df_copy = data.copy()
df_copy.loc[:, df_copy.columns != 'Move'] = df_copy.loc[:, df_copy.columns != 'Move'].replace([1,2], [2,1])

data = data.append(df_copy)
data.to_csv('data_e.csv', index=False)