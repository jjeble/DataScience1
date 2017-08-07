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




from sklearn.preprocessing import LabelEncoder
var_mod = list(df)
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  print(predictions)
  print(data[outcome])
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 


outcome_var = 'Major'
model = LogisticRegression()
predictor_var = ['Men']
classification_model(model, df,predictor_var,outcome_var)





plt.show()

