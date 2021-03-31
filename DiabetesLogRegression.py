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
# https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
from sklearn.model_selection import train_test_split    # splitting the data
from sklearn.linear_model import LogisticRegression     # LR algo
from sklearn.metrics import accuracy_score              # model precision

#-------------------------------------------------------------#
# Import the csv
#-------------------------------------------------------------#
df = pd.read_csv('C:\pytemp\diabetes\diabetes.csv')

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
# Scale values
# https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
#-------------------------------------------------------------#

df["Pregnancies"] = MinMaxScaler().fit_transform(np.array(df["Pregnancies"]).reshape(-1,1))
df["Glucose"] = MinMaxScaler().fit_transform(np.array(df["Glucose"]).reshape(-1,1))
df["BloodPressure"] = MinMaxScaler().fit_transform(np.array(df["BloodPressure"]).reshape(-1,1))
df["SkinThickness"] = MinMaxScaler().fit_transform(np.array(df["SkinThickness"]).reshape(-1,1))
df["Insulin"] = MinMaxScaler().fit_transform(np.array(df["Insulin"]).reshape(-1,1))
df["BMI"] = MinMaxScaler().fit_transform(np.array(df["BMI"]).reshape(-1,1))
df["DiabetesPedigreeFunction"] = MinMaxScaler().fit_transform(np.array(df["DiabetesPedigreeFunction"]).reshape(-1,1))
df["Age"] = MinMaxScaler().fit_transform(np.array(df["Age"]).reshape(-1,1))

#-------------------------------------------------------------#
# separate df of independent and dependent variables
#-------------------------------------------------------------#

X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

#-------------------------------------------------------------#
# done another way using variable names explicitly
#-------------------------------------------------------------#
X_var = df[['Pregnancies','Glucose','BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values # independent variable
y_var = df['Outcome'].values # dependent variable

#-------------------------------------------------------------#
# Show the variable samples
#-------------------------------------------------------------#
print(cl('X variable samples : {}'.format(X_var[:5]), attrs = ['bold']))
print(cl('Y variable samples : {}'.format(y_var[:5]), attrs = ['bold']))



#-------------------------------------------------------------#
# split df into training and test dfs
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#-------------------------------------------------------------#


x_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.3, random_state = 0)



#-------------------------------------------------------------#
#building logistic model
#-------------------------------------------------------------#
logmodel=LogisticRegression(solver='liblinear')
logmodel.fit(x_train,y_train)

#-------------------------------------------------------------#
#evaluating model performance
#-------------------------------------------------------------#
pred_model=logmodel.predict(X_test)

print(cl('Accuracy of the model is {:.0%}'.format(accuracy_score(y_test, pred_model)), attrs = ['bold']))
