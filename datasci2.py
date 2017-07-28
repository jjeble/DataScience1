from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



df = pd.read_csv("/Users/jayjeble/Documents/SZ/train.csv")

print(df.apply(lambda x: sum(x.isnull()),axis=0))

print(df['Gender'].value_counts())
print(df['Dependents'].value_counts())


df['Self_Employed'].fillna('No',inplace=True)
df['Gender'].fillna('Male',inplace=True)
df['Dependents'].fillna('0',inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(),inplace=True)
df['Married'].fillna('Yes',inplace=True)
df['Credit_History'].fillna(0,inplace=True)



table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns=['Gender'], aggfunc=np.median)
print(table)
def fage(x):
 return table.loc[x['Self_Employed'],x['Gender']]
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)






from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
print(df.apply(lambda x: sum(x.isnull()),axis=0))
for i in var_mod:
    print(i)
    df[i] = le.fit_transform(df[i])
print(df.dtypes)

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics



def classification_model(model,data,predictors,outcome):
    model.fit(data[predictors],data[outcome])

    predictions = model.predict(data[predictors])

    accuracy = metrics.accuracy_score(predictions,data[outcome])
    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    kf = KFold(data.shape[0],n_folds=5)
    error= []
    for train,test in kf:
        train_predictors = (data[predictors].iloc[train,:])
        train_target = data[outcome].iloc[train]
        model.fit(train_predictors,train_target)
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 

