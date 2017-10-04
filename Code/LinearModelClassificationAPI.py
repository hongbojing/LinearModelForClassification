import math
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel
from sklearn.metrics import log_loss

# scale larger positive and values to between -1,1 depending on the largest
# value in the data
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,8))
df = pd.read_csv("../../Dataset/data2.csv", header=0)


#15 4 5 6 7 10 14 16
#df.columns = ["lat","loc_x","loc_y","lon","time","shot_distance","shot_type"]
X = df[["lat","loc_x","loc_y","lon","time","shot_distance","shot_type"]]
X = np.array(X)

Y = df[["shot_made_flag"]]
Y = np.array(Y)


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)

clf = LogisticRegression()
clf.fit(X_train,Y_train)

realY = Y_test
predictY = clf.predict(X_test)

countMatch = 0
countNotMatch = 0
for i in range(realY.shape[0]):
    if(realY[i] == predictY[i]):
        countMatch = countMatch + 1
    else:
        countNotMatch = countNotMatch + 1

print("Total testing set : " + str(realY.shape[0]) )
print("predict Y matchs with real Y percentage : " + str(countMatch/realY.shape[0] ) )