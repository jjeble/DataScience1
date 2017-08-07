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

table = df.pivot_table(values='Men',index='Major_category',columns='Part_time')
print(table)



plt.show()

