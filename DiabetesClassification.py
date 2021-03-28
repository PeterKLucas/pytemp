import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('C:\pytemp\diabetes'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


dataset = pd.read_csv('C:\pytemp\diabetes\diabetes.csv')
print(f"shape of the Diabetes dataset :- {dataset.shape}")
print("\n ***************************** \n")
print(f"Sample Dataset :- \n {dataset.head()}")
## null values checking 
print("\n ***************************** \n")
print(f"checking for null values :- \n {dataset.isnull().sum()}")
print("\n ***************************** \n")
## checking for whether dataset have duplicate values or not
print(f"Number of Duplicate values :- {len(dataset.loc[dataset.duplicated()])}")


## separate dataset of independent and dependent variables
X = dataset.drop(['Outcome'], axis=1)
y = dataset['Outcome']

col_names = list(X.columns)
## craete pipe line with feature scaling
pipeline = Pipeline([
                     ('std_scale', PowerTransformer(method='yeo-johnson'))
])

X = pd.DataFrame(pipeline.fit_transform(X), columns=col_names)

print(X.head())
## split dataset into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.1, random_state=42)

print(f"Size Of The Train Dataset :- {len(X_train)}")
print(f"Size Of The Test Dataset :- {len(X_test)}")

train_scores = []
test_scores = []

for i in range(1, 25):
  knn_clf = KNeighborsClassifier(n_neighbors=i)
  knn_clf.fit(X_train, y_train)

  train_scores.append(knn_clf.score(X_train, y_train))
  test_scores.append(knn_clf.score(X_test, y_test))

print(f"Max score of Train dataset at K = {train_scores.index(max(train_scores)) + 1} and score :- {max(train_scores)*100}%")
print(f"Max score of Test dataset at K = {test_scores.index(max(test_scores)) + 1} and score :- {round(max(test_scores)*100, 2)}%")

## training history graph 
plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,25),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,25),test_scores,marker='o',label='Test Score')

## best score on test data at k = 5

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)

## predict X_test 
y_pred = knn_clf.predict(X_test)
print("\n ***************************** \n")
print(f"Accuracy :- \n {accuracy_score(y_test, y_pred)*100}")
print("\n ***************************** \n")
print(f"Confusion Matrix :- \n{confusion_matrix(y_test, y_pred)}")
print("\n ***************************** \n")
print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")