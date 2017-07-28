
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("/Users/jayjeble/Documents/SZ/train.csv")

print(df.apply(lambda x: sum(x.isnull()),axis=0))

print(df['Gender'].value_counts())
print(df['Dependents'].value_counts())


df['Self_Employed'].fillna('No',inplace=True)
df['Gender'].fillna('Male',inplace=True)
df['Dependents'].fillna(0,inplace=True)

table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns=['Gender'], aggfunc=np.median)
print(table)
def fage(x):
 return table.loc[x['Self_Employed'],x['Gender']]
##

df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
print(df.apply(lambda x: sum(x.isnull()),axis=0))
plt.show()
