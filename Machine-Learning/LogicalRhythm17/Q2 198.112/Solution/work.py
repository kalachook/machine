#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split

def set_upX(X):
    #Change X to DataFrame
    X = pd.DataFrame(X,columns = ['TIMESTAMP','STARTING_LATITUDE','STARTING_LONGITUDE','DESTINATION_LATITUDE','DESTINATION_LONGITUDE','VEHICLE_TYPE','TOTAL_LUGGAGE_WEIGHT','WAIT_TIME','TRAFFIC_STUCK_TIME','DISTANCE'])
    
    #Featch City
    chukk = []
    for i in range(20000):
        if ( (int(X['STARTING_LATITUDE'][i])) == 28 or (int(X['DESTINATION_LATITUDE'][i])) == 28 or (int(X['STARTING_LONGITUDE'][i])) == 76 or (int(X['DESTINATION_LONGITUDE'][i])) == 76 or ( (int(X['STARTING_LATITUDE'][i])) == 28 and (int(X['STARTING_LONGITUDE'][i])) == 77 ) or ( (int(X['STARTING_LATITUDE'][i])) == 28 and (int(X['STARTING_LONGITUDE'][i])) == 76 ) or ( (int(X['DESTINATION_LATITUDE'][i])) == 28 and (int(X['DESTINATION_LONGITUDE'][i])) == 77 ) or ((int(X['DESTINATION_LATITUDE'][i])) == 28 and (int(X['DESTINATION_LONGITUDE'][i])) == 76) ):
            chukk.append(1)
        elif (int(X['STARTING_LATITUDE'][i])) == 25 or (int(X['DESTINATION_LATITUDE'][i])) == 25 or (int(X['STARTING_LONGITUDE'][i])) == 82 or (int(X['STARTING_LONGITUDE'][i])) == 83 or (int(X['DESTINATION_LONGITUDE'][i])) == 82 or (int(X['DESTINATION_LONGITUDE'][i])) == 83:
            chukk.append(2)
        elif (int(X['STARTING_LATITUDE'][i])) == 22 or (int(X['DESTINATION_LATITUDE'][i])) == 22 or (int(X['STARTING_LONGITUDE'][i])) == 88 or (int(X['DESTINATION_LONGITUDE'][i])) == 88:
            chukk.append(3)
        elif (int(X['STARTING_LATITUDE'][i])) == 19 or (int(X['DESTINATION_LATITUDE'][i])) == 19 or (int(X['STARTING_LONGITUDE'][i])) == 72 or (int(X['DESTINATION_LONGITUDE'][i])) == 72:
            chukk.append(4)
        elif (int(X['STARTING_LONGITUDE'][i])) == 80 or (int(X['DESTINATION_LONGITUDE'][i])) == 80 or ( (int(X['STARTING_LATITUDE'][i])) == 12 and (int(X['STARTING_LONGITUDE'][i])) == 80 ) or ( (int(X['STARTING_LATITUDE'][i])) == 13 and (int(X['STARTING_LONGITUDE'][i])) == 80 ) or ( (int(X['DESTINATION_LATITUDE'][i])) == 12 and (int(X['DESTINATION_LONGITUDE'][i])) == 80 ) or ( (int(X['DESTINATION_LATITUDE'][i])) == 13 and (int(X['DESTINATION_LONGITUDE'][i])) == 80 ):
            chukk.append(5)
        elif (int(X['STARTING_LONGITUDE'][i])) == 77 or (int(X['DESTINATION_LONGITUDE'][i])) == 77 or ( (int(X['STARTING_LATITUDE'][i])) == 12 and (int(X['STARTING_LONGITUDE'][i])) == 77 ) or ( (int(X['STARTING_LATITUDE'][i])) == 13 and (int(X['STARTING_LONGITUDE'][i])) == 77 ) or ( (int(X['DESTINATION_LATITUDE'][i])) == 12 and (int(X['DESTINATION_LONGITUDE'][i])) == 77 ) or ( (int(X['DESTINATION_LATITUDE'][i])) == 13 and (int(X['DESTINATION_LONGITUDE'][i])) == 77 ):
            chukk.append(6)
        else:
            chukk.append(0)
     
    chukk = pd.DataFrame(chukk,columns = ['City'])
     
    #Delete Starting and Destination coordinates
    X.drop('STARTING_LATITUDE',axis=1,inplace=True)
    X.drop('STARTING_LONGITUDE',axis=1,inplace=True)
    X.drop('DESTINATION_LATITUDE',axis=1,inplace=True)
    X.drop('DESTINATION_LONGITUDE',axis=1,inplace=True)
    
    #Concatinate Location to main DataFrame
    X = pd.concat([X,chukk], axis=1)
     
    #Deal with dates
    #Split Date into Year, Month, Day, Hour, Minutes, Seconds
    temp = pd.DataFrame(X.TIMESTAMP.str.split().tolist(), columns="DATE TIME".split())
    temp2 = pd.DataFrame(temp.DATE.str.split('-').tolist(), columns="YEAR MONTH DAY".split())
    temp3 = pd.DataFrame(temp.TIME.str.split(':').tolist(), columns="HOUR MINUTS SECONDS".split())
     
    #Concatinate Year, Month, Day, Hour, Minutes, Seconds
    result = pd.concat([temp2,temp3],axis=1)
    X = pd.concat([X,result], axis=1)
    
    #Delete TimeStamp column
    X.drop('TIMESTAMP',axis=1,inplace=True)
     
     
    #Deal with Vechile
    #Change all Vechile to UpperCase
    chukk = []
    for i in range(20000):
        chukk.append(X['VEHICLE_TYPE'][i].upper())
    
    #Replace with UpderCase column
    X['VEHICLE_TYPE'] = chukk
    
    labelencoder = LabelEncoder()
    result = X.iloc[:,:].values
    result[:, 0] = labelencoder.fit_transform(result[:, 0])
    
    onehotencoder = OneHotEncoder(categorical_features = [7])
    result = onehotencoder.fit_transform(result).toarray()
    result = result[:,1:]
    result = result.astype(float)
     
    X = result
    return X
 
#Importing datasets
dataset = pd.read_csv('train.csv')

#Filling NaN and split
dataset = dataset.fillna(-1)
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Setting up X
X=set_upX(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
 
#Fitting RandomForestRegressor to Training set
reg = RandomForestRegressor(random_state=0,max_features=0.85,min_samples_leaf=9,n_estimators=685,n_jobs=-1)
reg.fit(X_train, y_train)

#Predict the Test set results
y_pred = reg.predict(X_test)

#Accuracy check
print(r2_score(y_test, y_pred)*200)



#Answer
#Importing datasets
Xans=pd.read_csv('test.csv')
Xans = Xans.fillna(-1)
Xans=Xans.iloc[:,1:].values
Xans=set_upX(Xans)

#Predict the Test set results
y_pred2=reg.predict(Xans)

#To save as csv
y_pred2.tofile('foo.csv', sep='\n', format='%10.6f')