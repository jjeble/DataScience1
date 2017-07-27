import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("/Users/jayjeble/Documents/SZ/train.csv")
df['ApplicantIncome'].hist(bins=50)
plt.show()
