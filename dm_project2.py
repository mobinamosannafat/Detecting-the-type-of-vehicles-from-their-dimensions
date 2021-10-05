import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Using pipeline to test different classifiers
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
# For comparing learning results and accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Read dataset from uploaded file and display it
dataset = pd.read_csv('cars.csv')
dataset

# Display dataset information
dataset.info()

# Seperating featues from class and saving them to numpy arrays
X = dataset[['x1' , 'y1' , 'x2' , 'y2']].values
Y = dataset[['type']].values
Y

#scatter plot
Y1 = np.zeros(Y.size)
for k in range(1,10053):
  if Y[(k)] == 'bus':
    Y1[(k)] = 0
  elif Y[(k)] == 'microbus':
    Y1[(k)] = 1
  elif Y[(k)] == 'minivan':
    Y1[(k)] = 2
  elif Y[(k)] == 'sedan':
    Y1[(k)] = 3
  elif Y[(k)] == 'suv':
    Y1[(k)] = 4
  else:
    Y1[(k)] = 5

X1 = X[:,2] - X[:,0]
X2 = X[:,3] - X[:,1]
fig , ax = plt.subplots(figsize=(20,14))

scatter = ax.scatter(X1, X2 , c=Y1)
legend1 = ax.legend(*scatter.legend_elements(num=5),
                    loc="upper left", title="class")
ax.add_artist(legend1)

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2)

# Using pipeline to test different classifiers
classifiers = [
               Pipeline([
                 ('scaler', StandardScaler()),
                 ('Logistic Regression', LogisticRegression())]),
               Pipeline([
                 ('scaler', StandardScaler()),
                 ('K_Neighbors Classifier', KNeighborsClassifier())]),
               Pipeline([
                 ('scaler', StandardScaler()),
                 ('Radius_Neighbors Classifier', RadiusNeighborsClassifier())]),
               Pipeline([
                 ('scaler', StandardScaler()),
                 ('Decision_Tree Classifier', DecisionTreeClassifier())]),
               Pipeline([
                 ('scaler', StandardScaler()),
                 ('Random_Forest Classifier', RandomForestClassifier())]),
               Pipeline([
                 ('scaler', StandardScaler()),
                 ('LinearSVC', LinearSVC())])
               ]

#name of classifiers
classifier_names = np.array(['Logistic Regression','KNeighborsClassifier','Radius_Neighbors Classifier', 'Decision_Tree Classifier','Random_Forest Classifier','LinearSVC'])

# For comparing learning results and accuracy

i=0;
for clf in classifiers:
  clf.fit(X_train, Y_train)
  Y_pred = clf.predict(X_test);
  print("\n\n<< ", classifier_names[i]," >>")
  print("\nAccracy : ", accuracy_score(Y_test, Y_pred))
  print("\nConfusion Matrix:\n",confusion_matrix(Y_test, Y_pred))
  print("\nReports:\n",classification_report(Y_test, Y_pred))
  print("\n--------------------------------------------------------------")
  i+=1;