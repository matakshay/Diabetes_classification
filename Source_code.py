#!/usr/bin/env python
# coding: utf-8
#Author: Akshay Mattoo

#import the requisite libraries
import numpy as np
import matplotlib as plot
import pandas as pd


#Read and store the Training Data
df1 = pd.read_csv("Diabetes_XTrain.csv")
df2 = pd.read_csv("Diabetes_YTrain.csv")
data1 = df1.values
Y_train = df2.values


#Remove the column headings
X_train = data1[1:,:]


#Calculate the distance between points by either of these methods
def car_distance (x1, x2): #Cartesian Distance
    return np.sqrt(sum((x1-x2)**2))

def man_distance (x1, x2): #Manhatten Distance
    return sum(abs(x1-x2))

def cheb_distance (x1, x2): #Chebyshev Distance
    return max(abs(x1-x2))


def kNN (X, Y, test_point, k):
    vals = []
    m = X.shape[0]
    
    for i in range (m):
        dist = car_distance(test_point, X[i]) #Calculate the distance from each training point
        vals.append((dist, Y[i]))
        
    vals = sorted(vals) #Sort the training points in increasing order of distance from test_point
    vals = vals[:k] #Take the k nearest points
    
    vals = np.array(vals) #Convert into a Numpy array

    new_vals = np.unique(vals[:,1], return_counts=True)
    
    index = new_vals[1].argmax() #Find the most commong group category of the k nearest points
    prediction = new_vals[0][index]
    
    return prediction



df3 = pd.read_csv("Diabetes_Xtest.csv") #Read test data
X_test = df3.values


with open ("result.csv",'w') as f: #Create a CSV file and store the prediction for each test point
    f.write("Outcome")
    f.write('\n')
    for i in range (0, 192, 1):
        pred = kNN(X_train, Y_train, X_test[i], 30)
        answer = str(pred)
        answer = answer[1:-1]
        answer += '\n'
        f.write(answer)


