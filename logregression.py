import pandas as pd                             # data processing
import numpy as np                              # working with arrays
import matplotlib.pyplot as plt                 # visualization
from matplotlib import rcParams                 # figure size
from termcolor import colored as cl             # text customization
from sklearn.preprocessing import MinMaxScaler  # pre processing


# https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/

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

# Scale values


dataset["Pregnancies"] = MinMaxScaler().fit_transform(np.array(dataset["Pregnancies"]).reshape(-1,1))
dataset["Glucose"] = MinMaxScaler().fit_transform(np.array(dataset["Glucose"]).reshape(-1,1))
dataset["BloodPressure"] = MinMaxScaler().fit_transform(np.array(dataset["BloodPressure"]).reshape(-1,1))
dataset["SkinThickness"] = MinMaxScaler().fit_transform(np.array(dataset["SkinThickness"]).reshape(-1,1))
dataset["Insulin"] = MinMaxScaler().fit_transform(np.array(dataset["Insulin"]).reshape(-1,1))
dataset["BMI"] = MinMaxScaler().fit_transform(np.array(dataset["BMI"]).reshape(-1,1))
dataset["DiabetesPedigreeFunction"] = MinMaxScaler().fit_transform(np.array(dataset["DiabetesPedigreeFunction"]).reshape(-1,1))
dataset["Age"] = MinMaxScaler().fit_transform(np.array(dataset["Age"]).reshape(-1,1))


#Transform data from numeric to categorical

for i in dataset.Outcome.values:
    if i  == 1:
        dataset.Outcome.replace(i, 'TRUE', inplace = True)
    else:
        dataset.Outcome.replace(i, 'FALSE', inplace = True)


## separate dataset of independent and dependent variables
X = dataset.drop(['Outcome'], axis=1)
y = dataset['Outcome']

## done another way using variable names explicitly
X_var = dataset[['Pregnancies','Glucose','BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values # independent variable
y_var = dataset['Outcome'].values # dependent variable

# Show the variable samples
print(cl('X variable samples : {}'.format(X_var[:5]), attrs = ['bold']))
print(cl('Y variable samples : {}'.format(y_var[:5]), attrs = ['bold']))




# split dataset into train and test datasets
from sklearn.model_selection import train_test_split    # splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.3, random_state = 0)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score              # model precision


#building logistic model
logmodel=LogisticRegression(solver='liblinear')
logmodel.fit(X_train,y_train)

#evaluating model performance
pred_model=logmodel.predict(X_test)

print(cl('Accuracy of the model is {:.0%}'.format(accuracy_score(y_test, pred_model)), attrs = ['bold']))
