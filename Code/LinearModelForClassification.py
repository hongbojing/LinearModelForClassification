import csv
import numpy as np
import random
from numpy import *
from sympy import E
from pylab import scatter, show, legend, xlabel, ylabel 

class GradientDescent:
    
    def sigmoid(self, X):   
        den =1.0 + exp(X) 
        gz =1.0/ den  
        return gz  
    
    def gradientFun(self, W, X, y, alpha, maxIterations):
        m = X.shape[1]
        xTrains = X.transpose()
        for i in range(maxIterations):
            h = self.sigmoid(np.dot(X, W))
            loss = h - y
            gradient = np.dot(xTrains, loss) / m
            W = W - alpha * gradient
        return W  

class Folds:    
    def initAFold(self, kFolds, sizeOfAFold):
        self.sizeOfAFold = sizeOfAFold
        self.data = np.ndarray((kFolds, sizeOfAFold), dtype=object)
    def setFoldValue(self, foldNum, num, data):
        self.data[foldNum, num] = data
    def getFoldValue(self, foldNum):
        return self.data[foldNum]
    
class Dataset:
    
    def __init__ (self, path, cols, rowNum):
        ifile = open(path, "r")
        reader = csv.reader(ifile)
        self.m_data = np.ndarray( (cols.shape[0], rowNum), dtype=object )
        row_count = 0
        for row in reader:
            col_count = 0
            for col in row:
        
                if row_count != 0:
                        for i in range(cols.shape[0]):
                            if col_count == cols[i]:
                                self.m_data[i][row_count - 1] = col
                col_count = col_count + 1
            row_count = row_count + 1
        ifile.close()
        #print("m_data")
        #print(self.m_data)

        

    def splitData(self, kFolds, sizeOfTestingData):
        self.sizeOfAFold = (int)((self.m_data.shape[1] - sizeOfTestingData) / kFolds)
        self.foldsList = [Folds() for i in range(self.m_data.shape[0])]
        self.testingDataSet = np.ndarray((self.m_data.shape[0], sizeOfTestingData), dtype=object)
        for i in range(self.m_data.shape[0]):
            self.foldsList[i].initAFold(kFolds, self.sizeOfAFold)
        
        for i in range(self.m_data.shape[0]):
            for j in range(sizeOfTestingData):
                self.testingDataSet[i][j] = self.m_data[i, j + self.sizeOfAFold * kFolds]  
        
        fold_count = 0
        for i in range(self.m_data.shape[0]):
            data_count = 0
            fold_count = 0
            for j in range(self.sizeOfAFold * kFolds):
                if data_count == self.sizeOfAFold:
                    fold_count = fold_count + 1
                    data_count = 0
                self.foldsList[i].setFoldValue(fold_count, data_count, self.m_data[i, j])
                data_count = data_count + 1
    
    def getFoldXData(self, col, foldNum):
        return self.foldsList[col].getFoldValue(foldNum)   

    def getTestingData(self, col):
        return self.testingDataSet[col, 0:]     

#------------------------------------------------------------------------------
m_alpha = 0.0000001
m_dataRaw = 25697
m_testingDataNum = 5140
m_foldsNum = 5
m_maxIterations = 1
m_cols = np.array( [15,5,6,7,10,4,14,16] )
m_dataset = Dataset('../../Dataset/data2.csv', m_cols, m_dataRaw)
#------------------------------------------------------------------------------

m_dataset.splitData(m_foldsNum, m_testingDataNum)

trX0 = np.ones((int)((m_dataRaw - m_testingDataNum) / m_foldsNum) * (m_foldsNum - 1) )

avgW = np.zeros( (m_cols.shape[0]) )

m_gradFunction = GradientDescent()

# gradient decent
for i in range(m_foldsNum):
    W = np.ones( (m_cols.shape[0]) )
    tempX = np.zeros((m_cols.shape[0] - 1, (int)((m_dataRaw - m_testingDataNum) / m_foldsNum) * (m_foldsNum - 1) ))    
    tempY = np.zeros((1, (int)((m_dataRaw - m_testingDataNum) / m_foldsNum) * (m_foldsNum - 1) ))    
    temp = np.zeros((m_cols.shape[0], (int)((m_dataRaw - m_testingDataNum) / m_foldsNum) * (m_foldsNum - 1) ))
    
    for j in range(m_cols.shape[0]):
        count = 0
        for k in range(m_foldsNum):
            #print(str(j) + ", " + str(k))
            if i != k:
                for l in range( (int)((m_dataRaw - m_testingDataNum) / m_foldsNum) ):
                    temp[j, count] = m_dataset.getFoldXData(j, k)[l]
                    count = count + 1

    for j in range(m_cols.shape[0]):
        for k in range((int)((m_dataRaw - m_testingDataNum) / m_foldsNum) * (m_foldsNum - 1)):
            if j == 0:
                tempY[j][k] = temp[j][k]
            else:
                tempX[j - 1][k] = temp[j][k]

    X = np.ones((m_cols.shape[0],(int)((m_dataRaw - m_testingDataNum) / m_foldsNum) * (m_foldsNum - 1)))
    for j in range(X.shape[0]):
        if j == 0:
            X[0] = trX0
        else:
            X[j] = tempX[j - 1]

    Y = np.ones( ((int)((m_dataRaw - m_testingDataNum) / m_foldsNum) * (m_foldsNum - 1))) 
    Y = tempY[0]
    W = m_gradFunction.gradientFun(W, X.T, Y.T, m_alpha, m_maxIterations)
    #print(W)
    avgW = avgW + W
avgW = avgW / 5
print("Average W")
print(avgW)

# Using testing set to prove Y value
tempTeData = np.zeros((m_cols.shape[0], m_testingDataNum))
for i in range(m_cols.shape[0]):
    tempTeData[i] = m_dataset.getTestingData(i)

teX0 = np.ones(m_testingDataNum)
predictY = np.zeros(m_testingDataNum)
for i in range(m_cols.shape[0] - 1):
    for j in range(m_testingDataNum):
        if i == 0:
            predictY[j] = predictY[j] + (teX0[j] * avgW[i])
        else:
            predictY[j] = predictY[j] + (tempTeData[i, j] * avgW[i])      
print("iterations = " + str(m_maxIterations) )
print("alpha = " + str(m_alpha) )
print("predict y")
print(predictY)
print("real y")
print(tempTeData[0])


temp = np.ndarray( (2, rowNum), dtype=object)
temp[0] = self.m_data[2, :]
temp[1] = self.m_data[5, :]

X = temp[0:2, 0:10000]
y = self.m_data[0, 0:10000]  
for i in range(10000):
    if y[i] == '1':
        scatter(X[0, i], X[1, i], marker='o', c='b')  
    else:
        scatter(X[0, i], X[1, i], marker='x', c='r')               
xlabel('loc x') 
ylabel('distance')  
#legend(['Fail', 'Shot made'])  
show()
