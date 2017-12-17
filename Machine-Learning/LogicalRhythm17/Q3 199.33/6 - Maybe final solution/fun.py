#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

def prepare(X):
    #Change X to DataFrame
    X = pd.DataFrame(X,columns = ['Build_Date','Priced_Date','Garden_Space','Dock','Capital','Royal','Guarding','River','Renovation','Dining_Rooms','Bedromms','Bathrooms','Visit','Sorcerer','Blessings','In_Front','Location','Holy_Tree','Distance_from_Knight','Builder'])

    ######## Start Dealing with Dates ########
    
    #Split Build_Date
    temp = pd.DataFrame(X.Build_Date.str.split().tolist(), columns="Date Time Meridiem".split())
    temp2 = pd.DataFrame(temp.Date.str.split('/').tolist(), columns="Build_Month Build_Day Build_Year".split())
    temp3 = pd.DataFrame(temp.Time.str.split(':').tolist(), columns="Build_Hour Build_Minutes".split())
    temp4 = pd.DataFrame(temp.iloc[:,-1].values, columns = ['Build_Meridiem'])
    
    #Concatinate Date,Time and Meridiem
    result = pd.concat([temp2,temp3,temp4],axis=1)
    
    #Label Encoding Build_Date
    labelencoder = LabelEncoder()
    test = result.iloc[:,:].values
    test[:,5] = labelencoder.fit_transform(test[:, 5])
    test = pd.DataFrame(test,columns=['Build_Month','Build_Day','Build_Year','Build_Hour','Build_Minutes','Build_Meridiem'])
    test['Build_Hour'] = test['Build_Hour'].astype(int)
    test['Build_Meridiem'] = test['Build_Meridiem'].astype(int)
    chukk = []
    for i in range(test.shape[0]):
        if test['Build_Meridiem'][i] == 1 :
            chukk.append(test['Build_Hour'][i] + 12)
        else:
            chukk.append(test['Build_Hour'][i])
    test['Build_Hour'] = chukk
    X = pd.concat([X,test], axis=1)
    X.drop('Build_Date',axis=1,inplace=True)
    
    #Split Priced_Date
    temp = pd.DataFrame(X.Priced_Date.str.split().tolist(), columns="Date Time Meridiem".split())
    temp2 = pd.DataFrame(temp.Date.str.split('/').tolist(), columns="Priced_Month Priced_Day Priced_Year".split())
    temp3 = pd.DataFrame(temp.Time.str.split(':').tolist(), columns="Priced_Hour Priced_Minutes".split())
    temp4 = pd.DataFrame(temp.iloc[:,-1].values, columns = ['Priced_Meridiem'])
    
    #Concatinate Date,Time and Meridiem
    result = pd.concat([temp2,temp3,temp4],axis=1)
    
    #Label Encoding Priced_Date
    labelencoder = LabelEncoder()
    test = result.iloc[:,:].values
    test[:,5] = labelencoder.fit_transform(test[:, 5])
    test = pd.DataFrame(test,columns=['Priced_Month','Priced_Day','Priced_Year','Priced_Hour','Priced_Minutes','Priced_Meridiem'])
    test['Priced_Hour'] = test['Priced_Hour'].astype(int)
    test['Priced_Meridiem'] = test['Priced_Meridiem'].astype(int)
    chukk = []
    for i in range(test.shape[0]):
        if test['Priced_Meridiem'][i] == 1 :
            chukk.append(test['Priced_Hour'][i] + 12)
        else:
            chukk.append(test['Priced_Hour'][i])
    test['Priced_Hour'] = chukk
    X = pd.concat([X,test], axis=1)
    X.drop('Priced_Date',axis=1,inplace=True)

    ######## End Dealing with Dates ########
    
    X = (X.iloc[:,:].values).astype(float)
    
    # Encoding categorical data
    labelencoder_X = LabelEncoder()
    
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    onehotencoder = OneHotEncoder(categorical_features = [0])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:,1:]
    
    X[:, 7] = labelencoder_X.fit_transform(X[:, 7])
    onehotencoder = OneHotEncoder(categorical_features = [7])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:,1:]
    
    X[:, 12] = labelencoder_X.fit_transform(X[:, 12])
    onehotencoder = OneHotEncoder(categorical_features = [12])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:,1:]
    
    X[:, 15] = labelencoder_X.fit_transform(X[:, 15])
    onehotencoder = OneHotEncoder(categorical_features = [15])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:,1:]
    
    X[:, 17] = labelencoder_X.fit_transform(X[:, 17])
    onehotencoder = OneHotEncoder(categorical_features = [17])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:,1:]
    
    X[:, 21] = labelencoder_X.fit_transform(X[:, 21])
    onehotencoder = OneHotEncoder(categorical_features = [21])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:,1:]
    
    X[:, 24] = labelencoder_X.fit_transform(X[:, 24])
    onehotencoder = OneHotEncoder(categorical_features = [24])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:,1:]
    
    X[:, 31] = labelencoder_X.fit_transform(X[:, 31])
    onehotencoder = OneHotEncoder(categorical_features = [31])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:,1:]
        
    return X

#Importing datasets
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Taking care of missing data
imputer = Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
imputer = imputer.fit(X[:,2:])
X[:,2:] = imputer.transform(X[:,2:])

#Prepare X
X = prepare(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Linear Regression to Training set
regressor = LinearRegression(fit_intercept = True)
regressor.fit(X_train, y_train)

#Predict the Test set results
y_pred = regressor.predict(X)

#Accuracy check
print(r2_score(y, y_pred)*200)



##Answer
#Importing datasets
dataset = pd.read_csv('test.csv')
X = dataset.iloc[:, 1:].values
X[:,2:] = imputer.transform(X[:,2:])
X = prepare(X)

#Predict the Test set results
y_pred2 = regressor.predict(X)

#To save as csv
y_pred2.tofile('foo2.csv', sep='\n', format='%10.0f')