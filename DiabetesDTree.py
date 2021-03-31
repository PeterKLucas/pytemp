#-------------------------------------------------------------#
# 
#
# Create and evaluate a DecisionTree based model regarding
# predicting diabetes based on 8 input variables
#-------------------------------------------------------------#


#-------------------------------------------------------------#
# imports
#-------------------------------------------------------------#
import pandas as pd                 # data processing
import numpy as np                  # working with arrays
import matplotlib.pyplot as plt     # visualization
from matplotlib import rcParams     # figure size
from termcolor import colored as cl # text customization


#-------------------------------------------------------------#
# scikit-learn
#-------------------------------------------------------------#
# Machine Learning in Python
# Simple and efficient tools for data mining and data analysis
#-------------------------------------------------------------#

from sklearn.tree import DecisionTreeClassifier as dtc          # alg
from sklearn.model_selection import train_test_split            # splitting the data
from sklearn.metrics import accuracy_score                      # model accuracy
from sklearn.tree import plot_tree                              # tree diagram

#-------------------------------------------------------------#
# List files in working folder
#-------------------------------------------------------------#

import os
for dirname, _, filenames in os.walk('C:\pytemp\diabetes'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#-------------------------------------------------------------#
# Import the csv
#-------------------------------------------------------------#
df = pd.read_csv('C:\pytemp\diabetes\diabetes.csv')

#-------------------------------------------------------------#
# Explore and prepare diabetes dataset 
#-------------------------------------------------------------#

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
# null values checking 
#-------------------------------------------------------------#
print(f"null count :- \n {df.isnull().sum()}\n")

#-------------------------------------------------------------#
# check for dupes
#-------------------------------------------------------------#
print(f"dupes  :- {len(df.loc[df.duplicated()])}")

#-------------------------------------------------------------#
#replace numerical variable with category (for DTree)
#-------------------------------------------------------------#
for i in df.Outcome.values:
    if i  == 1:
        df.Outcome.replace(i, 'TRUE', inplace = True)
    else:
        df.Outcome.replace(i, 'FALSE', inplace = True)

#-------------------------------------------------------------#
# separate df into independent and dependent variables
#-------------------------------------------------------------#
X_var = df.drop(['Outcome'], axis=1)  #independent
y_var = df['Outcome']  #dependent


#-------------------------------------------------------------#
## done another way using variable names explicitly
#-------------------------------------------------------------#

#X_var = df[['Pregnancies','Glucose','BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values # independent variable
#y_var = df['Outcome'].values # dependent variable


print(cl('X variable samples : {}'.format(X_var[:5]), attrs = ['bold']))
print(cl('Y variable samples : {}'.format(y_var[:5]), attrs = ['bold']))

#-------------------------------------------------------------#
# split df into training and test dfs
#-------------------------------------------------------------#
x_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.3, random_state = 0)

#-------------------------------------------------------------#
# instantiate and fit model
#-------------------------------------------------------------#
model = dtc(criterion = 'entropy', max_depth = 4)
model.fit(x_train, y_train)

pred_model = model.predict(X_test)

#-------------------------------------------------------------#
# test for accuracy
#-------------------------------------------------------------#
print(cl('Accuracy is {:.0%}'.format(accuracy_score(y_test, pred_model)), attrs = ['bold']))

#-------------------------------------------------------------#
# plot results of decision tree
#-------------------------------------------------------------#

feature_names = df.columns[:10]
target_names = df['Outcome'].unique().tolist()

plot_tree(model, 
          feature_names = feature_names, 
          class_names = target_names, 
          filled = True, 
          rounded = True)

plt.show()
plt.savefig('tree_visualization.png') 