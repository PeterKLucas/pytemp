#-------------------------------------------------------------#
#
# Create and evaluate a RandomForest based model regarding
# predicting a stroke based on 10 input variables
#-------------------------------------------------------------#

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
from sklearn.preprocessing import LabelEncoder # for one hot
from sklearn.metrics import accuracy_score              # model precision

#-------------------------------------------------------------#
# Import the csv
#-------------------------------------------------------------#
df = pd.read_csv('C:\pytemp\stroke\healthcare-dataset-stroke-data.csv')
print(df)

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
# convert categorical via one hot encoding; required for RF
# https://stackabuse.com/one-hot-encoding-in-python-with-pandas-and-scikit-learn/#:~:text=What%20is%20One-Hot%20Encoding%3F%20One-hot%20Encoding%20is%20a,a%20boolean%20specifying%20a%20category%20of%20the%20element.
#-------------------------------------------------------------#

lb=LabelEncoder()
df["hypertension"]=lb.fit_transform(df["hypertension"])
df["heart_disease"]=lb.fit_transform(df["heart_disease"])
df["ever_married"]=lb.fit_transform(df["ever_married"])
df["work_type"]=lb.fit_transform(df["work_type"])
df["Residence_type"]=lb.fit_transform(df["Residence_type"])
df["smoking_status"]=lb.fit_transform(df["smoking_status"])
df["gender"]=lb.fit_transform(df["gender"])

#-------------------------------------------------------------#
# separate dataset of independent and dependent variables
#-------------------------------------------------------------#
x_var = df.drop(['stroke'], axis=1)
y_var = df['stroke']

#-------------------------------------------------------------#
# split dataset into train and test datasets
#-------------------------------------------------------------#
from sklearn.model_selection import train_test_split    # splitting the data
x_train, x_test, y_train, y_test = train_test_split(x_var, y_var, test_size = 0.3, random_state = 0)


#-------------------------------------------------------------#
# instantiate and fit model
#-------------------------------------------------------------#

model =RandomForestClassifier().fit(x_train,y_train)
pred_model =model.predict(x_test)
ac = accuracy_score(y_test,pred_model)


#-------------------------------------------------------------#
#evaluate model performance
#-------------------------------------------------------------#
print("RandomForestClassifier model accuary",ac)
print(cl('Accuracy of the model is {:.0%}'.format(accuracy_score(y_test, pred_model)), attrs = ['bold']))
