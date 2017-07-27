
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


fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit History")
temp1.plot(kind="bar")


ax2 = fig.add_subplot(122)
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")
temp2.plot(kind = 'bar')


plt.show()

