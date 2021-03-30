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




df = pd.read_csv('C:\pytemp\diabetes\diabetes.csv')

print(df.info())

print(f"shape of the Diabetes df :- {df.shape}")
print("\n ***************************** \n")
print(f"Sample df :- \n {df.head()}")
## null values checking 
print("\n ***************************** \n")
print(f"checking for null values :- \n {df.isnull().sum()}")
print("\n ***************************** \n")
## checking for whether df have duplicate values or not
print(f"Number of Duplicate values :- {len(df.loc[df.duplicated()])}")


for i in df.Outcome.values:
    if i  == 1:
        df.Outcome.replace(i, 'TRUE', inplace = True)
    else:
        df.Outcome.replace(i, 'FALSE', inplace = True)


## separate df of independent and dependent variables
X_var = df.drop(['Outcome'], axis=1)
y_var = df['Outcome']

print(X_var)

## done another way using variable names explicitly
#X_var = df[['Pregnancies','Glucose','BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values # independent variable
#y_var = df['Outcome'].values # dependent variable


print(cl('X variable samples : {}'.format(X_var[:5]), attrs = ['bold']))
print(cl('Y variable samples : {}'.format(y_var[:5]), attrs = ['bold']))

# split df into train and test dfs

X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.3, random_state = 0)



model = dtc(criterion = 'entropy', max_depth = 4)
model.fit(X_train, y_train)

pred_model = model.predict(X_test)

print(cl('Accuracy of the model is {:.0%}'.format(accuracy_score(y_test, pred_model)), attrs = ['bold']))


feature_names = df.columns[:10]
target_names = df['Outcome'].unique().tolist()

plot_tree(model, 
          feature_names = feature_names, 
          class_names = target_names, 
          filled = True, 
          rounded = True)

plt.show()
plt.savefig('tree_visualization.png') 