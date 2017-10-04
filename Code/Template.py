
import numpy as np
import random
from numpy import genfromtxt
 
def getData(dataSet):
    m, n = np.shape(dataSet)
    trainData = np.ones((m, n))
    trainData[:,:-1] = dataSet[:,:-1]
    trainLabel = dataSet[:,-1]
    return trainData, trainLabel
 
def batchGradientDescent(x, y, theta, alpha, m, maxIterations):
    print(x)
    xTrains = x.transpose() 
    for i in range(0, maxIterations):
        hypothesis = np.dot(x, theta)
        #print(hypothesis)
        loss = hypothesis - y
        gradient = np.dot(xTrains, loss) / m
        #print(gradient)
        theta = theta - alpha * gradient
        #print("")
    return theta
 
def predict(x, theta):
    m, n = np.shape(x)
    xTest = np.ones((m, n+1))
    xTest[:, :-1] = x
    yP = np.dot(xTest, theta)
    return yP
 
dataPath = r"../../Dataset/mdata.csv"
dataSet = genfromtxt(dataPath, delimiter=',')
trainData, trainLabel = getData(dataSet)
m, n = np.shape(trainData)
theta = np.ones(n)

alpha = 0.1
maxIteration = 5000
theta = batchGradientDescent(trainData, trainLabel, theta, alpha, m, maxIteration)
print(theta)
#x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])
print (predict(x, theta))