#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV

#Importing datasets
dataset = pd.read_csv("train.csv")
X = dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train ,y_test = train_test_split(X,Y,test_size = 0.2,random_state=0)

#Feature Scalling of training and testing data
scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

##After testing everything we used support vector classifier with RFECV##
clfsvm = SVC(kernel = 'linear')
selected=RFECV(clfsvm,n_jobs=-1)
selected.fit(X_train,y_train)
y_pred = selected.predict(X_test)

#Accuracy check
c = 0
for i in range(X_test.shape[0]):
    if y_pred[i] != y_test[i]:
        c = c + 1
print((X_test.shape[0] - c)/X_test.shape[0]*100)


#Answer
#Importing datasets
test_set= pd.read_csv("test.csv")
y_test=test_set.values
#Feature Scaling
y_test = scaler.transform(y_test)
#Predicting values
y_pred=selected.predict(y_test)
#Save to csv
df = pd.DataFrame(y_pred)
df.index = np.arange(1, len(df)+1)
df.to_csv("pred2.csv",sep=',')