# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import tree
dataset = pd.read_csv("/home/satyam/Downloads/MobilePricesClassification/train.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

#change test data  into two parts
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train ,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=30)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)

y_pred = classifier.predict(X_test)

#performance
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)
#decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,Y_train)
y1_pred = clf.predict(X_test)
cm1 = confusion_matrix(Y_test,y1_pred)
#kneighbours
from sklearn.neighbors import KNeighborsClassifier
clfknn = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p=2)
clfknn = clfknn.fit(X_test,Y_test)
y2_pred = clfknn.predict(X_test)

cm2 = confusion_matrix(Y_test,y2_pred)

from sklearn.svm import SVC
clfsvm = SVC(kernel = 'linear',degree = 3,random_state = 1)
clfsvm.fit(X_train,Y_train)
y3_pred = clfsvm.predict(X_test)

from sklearn.metrics import r2_score
r2 = r2_score(Y_test, y3_pred)*100

cm3 = confusion_matrix(Y_test,y3_pred)
demo = range(1,401)
plt.scatter(demo,Y_test)
plt.scatter(demo,y_pred,color = 'red')
plt.show()

            

#Final test
d2 = pd.read_csv('/home/satyam/Downloads/MobilePricesClassification/test.csv')
d2 = d2.fillna(-1)
X2 = d2.values
y_pred2 = clfsvm.predict(X2)

#To save as csv
y_pred2.tofile('/home/satyam/Downloads/MobilePricesClassification/foo.csv',sep='\n')