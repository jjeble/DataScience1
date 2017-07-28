
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("/Users/jayjeble/Documents/SZ/train.csv")

print(df.apply(lambda x: sum(x.isnull()),axis=0))

df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace = True)

print(df.apply(lambda x: sum(x.isnull()),axis=0))

df.boxplot(column="LoanAmount",by=["Education","Self_Employed"])

df['Self_Employed'].fillna('No',inplace=True)


plt.show()
