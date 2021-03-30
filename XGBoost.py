
import pandas as pd                             # data processing
import numpy as np                              # working with arrays
import matplotlib.pyplot as plt                 # visualization
from matplotlib import rcParams                 # figure size
from termcolor import colored as cl             # text customization
from sklearn.preprocessing import MinMaxScaler  # pre processing



## separate dataset of independent and dependent variables
X_var = dataset.drop(['Outcome'], axis=1)
y_var = dataset['Outcome']

df = pd.read_csv('C:\pytemp\stroke\healthcare-dataset-stroke-data.csv')
print(df)

# split dataset into train and test datasets
from sklearn.model_selection import train_test_split    # splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.3, random_state = 0)



from xgboost import XGBClassifier
xgclassifier = XGBClassifier()
xgclassifier.fit(X_train, y_train)

#evaluating model performance
pred_model=xgclassifier.predict(X_test)

print(cl('Accuracy of the model is {:.0%}'.format(accuracy_score(y_test, pred_model)), attrs = ['bold']))
