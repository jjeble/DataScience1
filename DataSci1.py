
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("/Users/jayjeble/Documents/SZ/train.csv")
temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values="Loan_Status",index=['Credit_History'],aggfunc = lambda x:x.map({'Y':1,'N':0}).mean())

print('Frequency table for credit history')
print(temp1)

print('\nProbility of getting loan for each Credit History class:')
print(temp2)
#plt.show()
