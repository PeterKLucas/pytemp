
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


#for i in dataset.Outcome.values:
 #   if i  == 'X':
      #  dataset.Outcome.replace(i, 'TRUE', inplace = True)
  #  else:
       # dataset.Outcome.replace(i, 'FALSE', inplace = True)


## separate dataset of independent and dependent variables
X = dataset.drop(['Outcome'], axis=1)
y = dataset['Outcome']

## done another way using variable names explicitly
X_var = dataset[['SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values # independent variable
y_var = dataset['Outcome'].values # dependent variable

print(cl('X variable samples : {}'.format(X_var[:5]), attrs = ['bold']))
print(cl('Y variable samples : {}'.format(y_var[:5]), attrs = ['bold']))

# split dataset into train and test datasets

x_train, x_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.3, random_state = 0)



from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score


models = []
#models.append(['Logistic Regreesion', LogisticRegression(random_state=0)])
#models.append(['SVM', SVC(random_state=0)])
models.append(['KNeighbors', KNeighborsClassifier()])
models.append(['GaussianNB', GaussianNB()])
models.append(['BernoulliNB', BernoulliNB()])
models.append(['Decision Tree', DecisionTreeClassifier(random_state=0)])
models.append(['Random Forest', RandomForestClassifier(random_state=0)])
#models.append(['XGBoost', XGBClassifier(eval_metric= 'error')])

lst_1= []

for m in range(len(models)):
    lst_2= []
    model = models[m][1]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)  #Confusion Matrix
    accuracies = cross_val_score(estimator = model, X = x_train, y = y_train, cv = 10)   #K-Fold Validation
    roc = roc_auc_score(y_test, y_pred)  #ROC AUC Score
    precision = precision_score(y_test, y_pred)  #Precision Score
    recall = recall_score(y_test, y_pred)  #Recall Score
    f1 = f1_score(y_test, y_pred)  #F1 Score
    print(models[m][0],':')
    print(cm)
    print('Accuracy Score: ',accuracy_score(y_test, y_pred))
    print('')
    print("K-Fold Validation Mean Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print('')
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
    print('')
    print('ROC AUC Score: {:.2f}'.format(roc))
    print('')
    print('Precision: {:.2f}'.format(precision))
    print('')
    print('Recall: {:.2f}'.format(recall))
    print('')
    print('F1: {:.2f}'.format(f1))
    print('-----------------------------------')
    print('')
    lst_2.append(models[m][0])
    lst_2.append((accuracy_score(y_test, y_pred))*100) 
    lst_2.append(accuracies.mean()*100)
    lst_2.append(accuracies.std()*100)
    lst_2.append(roc)
    lst_2.append(precision)
    lst_2.append(recall)
    lst_2.append(f1)
    lst_1.append(lst_2)