import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import requests
url="https://raw.githubusercontent.com/fivethirtyeight/data/master/college-majors/recent-grads.csv"
s=requests.get(url).content
df=pd.read_csv(io.StringIO(s.decode('utf-8')))

temp1 = df['Major_category'].value_counts(ascending = True)
temp2 = df.pivot_table(values = 'Unemployment_rate', index = ['Major_category'],aggfunc = lambda x: x.mean())

print(df.apply(lambda x:sum(x.isnull()),axis=0))
x = df.loc[(df['Major_category'] == 'Agriculture & Natural Resources') & (df['Men']!=np.nan),('Men')].sum()
y = df.loc[(df['Major_category'] == 'Agriculture & Natural Resources') & (df['Women']!=np.nan),('Women')].sum()

z = df.loc[(df['Major_category'] == 'Agriculture & Natural Resources') & (df['Men']!=np.nan)]


df['Men'].fillna(int(df['Men'].mean()), inplace=True)
df['Women'].fillna(int(df['Women'].mean()), inplace=True)
df['Total'].fillna(df['Men']+df['Women'], inplace=True)
df['ShareWomen'].fillna(df['Women']/df['Total'], inplace=True)

print(df.head(25))









plt.show()

