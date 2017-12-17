import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Ploting Scatter graph between STARTING_LATITUDE and STARTING_LONGITUDE for analysis
dataset = pd.read_csv('train.csv')
dataset.plot(kind="scatter", x="STARTING_LATITUDE", y="STARTING_LONGITUDE")

######Ploting Scatter graph between Distance and Fare by City

dataset = dataset.fillna(-1)
X = dataset.iloc[:,1:].values

#Change X to DataFrame
X = pd.DataFrame(X,columns = ['TIMESTAMP','STARTING_LATITUDE','STARTING_LONGITUDE','DESTINATION_LATITUDE','DESTINATION_LONGITUDE','VEHICLE_TYPE','TOTAL_LUGGAGE_WEIGHT','WAIT_TIME','TRAFFIC_STUCK_TIME','DISTANCE','FARE'])

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


import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
from matplotlib.colors import ListedColormap

sns.set(style="white", color_codes=True)

sns.FacetGrid(X, hue="City", size=5).map(plt.scatter, "DISTANCE", "FARE").add_legend()