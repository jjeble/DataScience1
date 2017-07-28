
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("/Users/jayjeble/Documents/SZ/train.csv")

print(df.apply(lambda x: sum(x.isnull()),axis=0))

df['Self_Employed'].fillna('No',inplace=True)

table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
print(df.apply(lambda x: sum(x.isnull()),axis=0))
plt.show()
