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
#temp2 = df.pivot_table(values = '
print(temp1)
print(temp2)

fig = plt.figure(figsize=(8,4))
ax1 =  fig.add_subplot(121)
ax1.set_xlabel('Major Category')
ax1.set_ylabel('Count of people')
#temp1.plot(kind = 'bar')

ax2 =  fig.add_subplot(122)
#temp2.plot(kind='bar')
ax2.set_xlabel('Category')
ax2.set_ylabel('Probablility of Unemployment')

temp3 = pd.crosstab(df['Major_category'],df['Unemployment_rate'])
print(temp3)






plt.show()

