#-------------------------------------------------------------#
# Create a list data structure containing several models
# Iterate over list and process input data, compare accuracy
#-------------------------------------------------------------#
import pandas as pd                 # data processing
import numpy as np                  # working with arrays
import matplotlib.pyplot as plt     # visualization
from matplotlib import rcParams     # figure size
from termcolor import colored as cl # text formatting

#-------------------------------------------------------------#
#scikit-learn
#-------------------------------------------------------------#
#Machine Learning in Python
#Simple and efficient tools for data mining and data analysis
#-------------------------------------------------------------#

from sklearn.tree import DecisionTreeClassifier as dtc  # tree algorithm
from sklearn.model_selection import train_test_split    # splitting the data
from sklearn.metrics import accuracy_score              # model precision
from sklearn.tree import plot_tree                      # tree diagram

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('C:\pytemp\diabetes'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#Import the CSV file

df = pd.read_csv('C:\pytemp\diabetes\diabetes.csv')

#-------------------------------------------------------------#
# data conversion 
#-------------------------------------------------------------#
#for i in df.Outcome.values:
 #   if i  == 'X':
      #  df.Outcome.replace(i, 'TRUE', inplace = True)
  #  else:
       # df.Outcome.replace(i, 'FALSE', inplace = True)

#-------------------------------------------------------------#
# separate df into independent and dependent variables
#-------------------------------------------------------------#
x_var= df.drop(['Outcome'], axis=1)
y_var = df['Outcome']

#-------------------------------------------------------------#
# done another way using variable names explicitly
#-------------------------------------------------------------#
x_var= df[['Pregnancies','Glucose','BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values # independent variable
y_var = df['Outcome'].values # dependent variable

print(cl('X variable samples : {}'.format(x_var[:5]), attrs = ['bold']))
print(cl('Y variable samples : {}'.format(y_var[:5]), attrs = ['bold']))

# split df into train and test dfs

x_train, x_test, y_train, y_test = train_test_split(x_var, y_var, test_size = 0.3, random_state = 0)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier   - could not install on Windows :( 
from sklearn.metrics import accuracy_score 


#-------------------------------------------------------------#
# create a list of model type
# append an instance of several compatible alg
#-------------------------------------------------------------#
models = []

models.append(['KNeighbors', KNeighborsClassifier()])
models.append(['GaussianNB', GaussianNB()])
models.append(['BernoulliNB', BernoulliNB()])
models.append(['Decision Tree', DecisionTreeClassifier(random_state=0)])
models.append(['Random Forest', RandomForestClassifier(random_state=0)])


#-------------------------------------------------------------#
# iterate over the list
# calculate accuracy for each model in list
#-------------------------------------------------------------#
for m in range(len(models)):
   
    model = models[m][1]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
  
    print('-------------------------------------')
    print(models[m][0],':')  #model name
    print('')
    print('Accuracy1 Score: ',accuracy_score(y_test, y_pred))
    print('')
    print('-------------------------------------')
    print('')
