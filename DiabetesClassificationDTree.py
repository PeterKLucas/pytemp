import pandas as pd                 # data processing
import numpy as np                  # working with arrays
import matplotlib.pyplot as plt     # visualization
from matplotlib import rcParams     # figure size
from termcolor import colored as cl # text customization

#scikit-learn
#------------
#Machine Learning in Python
#Simple and efficient tools for data mining and data analysis
#

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




dataset = pd.read_csv('C:\pytemp\diabetes\diabetes.csv')

print(dataset.info())

print(f"shape of the Diabetes dataset :- {dataset.shape}")
print("\n ***************************** \n")
print(f"Sample Dataset :- \n {dataset.head()}")
## null values checking 
print("\n ***************************** \n")
print(f"checking for null values :- \n {dataset.isnull().sum()}")
print("\n ***************************** \n")
## checking for whether dataset have duplicate values or not
print(f"Number of Duplicate values :- {len(dataset.loc[dataset.duplicated()])}")


for i in dataset.Outcome.values:
    if i  == 1:
        dataset.Outcome.replace(i, 'TRUE', inplace = True)
    else:
        dataset.Outcome.replace(i, 'FALSE', inplace = True)


## separate dataset of independent and dependent variables
X = dataset.drop(['Outcome'], axis=1)
y = dataset['Outcome']

## done another way using variable names explicitly
X_var = dataset[['SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values # independent variable
y_var = dataset['Outcome'].values # dependent variable

print(cl('X variable samples : {}'.format(X_var[:5]), attrs = ['bold']))
print(cl('Y variable samples : {}'.format(y_var[:5]), attrs = ['bold']))

# split dataset into train and test datasets

X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.3, random_state = 0)



model = dtc(criterion = 'entropy', max_depth = 4)
model.fit(X_train, y_train)

pred_model = model.predict(X_test)

print(cl('Accuracy of the model is {:.0%}'.format(accuracy_score(y_test, pred_model)), attrs = ['bold']))


feature_names = dataset.columns[:5]
target_names = dataset['Outcome'].unique().tolist()

plot_tree(model, 
          feature_names = feature_names, 
          class_names = target_names, 
          filled = True, 
          rounded = True)

plt.show()
plt.savefig('tree_visualization.png') 