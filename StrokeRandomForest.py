
#-------------------------------------------------------------#
# imports
#-------------------------------------------------------------#

import pandas as pd                             # data processing
import numpy as np                              # working with arrays
import matplotlib.pyplot as plt                 # visualization
from matplotlib import rcParams                 # figure size
from termcolor import colored as cl             # text customization

#-------------------------------------------------------------#
# scikit-learn
#-------------------------------------------------------------#
# Machine Learning in Python
# Simple and efficient tools for data mining and data analysis
#-------------------------------------------------------------#

from sklearn.preprocessing import MinMaxScaler  # pre processing
from sklearn.ensemble import RandomForestClassifier # alg


#-------------------------------------------------------------#
# Import the csv
#-------------------------------------------------------------#
df = pd.read_csv('C:\pytemp\stroke\healthcare-dataset-stroke-data.csv')
print(df)


print(df.info())
#-------------------------------------------------------------#
# check the shape
#-------------------------------------------------------------#
print(f"shape  df :- {df.shape}\n")

#-------------------------------------------------------------#
# sample 
#-------------------------------------------------------------#
print(f"Sample df :- \n {df.head()}\n")

#-------------------------------------------------------------#
# check for dupes
#-------------------------------------------------------------#
print(f"dupes  :- {len(df.loc[df.duplicated()])}")

#-------------------------------------------------------------#
# null values checking 
#-------------------------------------------------------------#
print(f"null count :- \n {df.isnull().sum()}\n")

#-------------------------------------------------------------#
# fix nulls by imputing mean value from bmi field
#-------------------------------------------------------------#
df["bmi"]=df["bmi"].fillna(df["bmi"].mean()) 

#-------------------------------------------------------------#
# null values re-check
#-------------------------------------------------------------#
print(f"null re-count :- \n {df.isnull().sum()}\n")


#-------------------------------------------------------------#
# separate dataset of independent and dependent variables
#-------------------------------------------------------------#
X_var = df.drop(['stroke'], axis=1)
y_var = df['stroke']

#-------------------------------------------------------------#
# split dataset into train and test datasets
#-------------------------------------------------------------#
from sklearn.model_selection import train_test_split    # splitting the data
x_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.3, random_state = 0)


#-------------------------------------------------------------#
# instantiate and fit model
#-------------------------------------------------------------#

model =RandomForestClassifier().fit(x_train,y_train)
y_pred =model.predict(x_test)
ac = accuracy_score(y_test,y_pred)
con = confusion_matrix(y_test, y_pred)
accuracies = []
accuracies.append(ac)
print("RandomForestClassifier model accuary",ac)
print(con)

print(cl('Accuracy of the model is {:.0%}'.format(accuracy_score(y_test, pred_model)), attrs = ['bold']))
