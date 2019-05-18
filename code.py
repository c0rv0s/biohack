#imports
import numpy as np
import matplotlib
matplotlib.use('TKAgg',warn=False, force=True)
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from collections import Counter
import pandas as pd
import random
import math
from numpy.random import choice
#end imports

#data import
df = pd.read_csv("heart.csv", usecols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13])
full_data = df.astype(float).values.tolist()

k=""
while( not k == "q"):
    k=input("enter a k value: ")
    if k == "q":
        break
    random.shuffle(full_data)

    #Question 1
    #method to calculate distance
    def distance(x,y,p):
        n = len(x)
        sum = 0
        for i in range(0,n):
            sum += abs(x[i]-y[i]) ** p
        return float('%.3f'%(sum ** (1.0/p)))

    #nearest neighbor classifier
    # the last value in each vector is assumed to be the correct answer
    def knn_classifier(x_test, x_train, y_train, k, p):
        y_pred = []
        for vector in x_test:
            closest = []
            vector = vector[:-1] #trim the class
            for trainee in x_train:
                result = trainee[-1] #save the result of this test
                dist = distance(vector, trainee[:-1], p)
                if len(closest) < k:
                    closest.append( (dist,result) )
                else:
                    farthest = closest[0]
                    index = 0
                    for i in range(0, k):
                        tup = closest[i]
                        if tup[0] > farthest[0]:
                            farthest = tup
                            index = i
                    closest[index] = (dist,result)
            #determine vector's closest neighbors results and add result to y_pred
            malig_count = 0.0
            for tup in closest:
                if tup[1] == y_train[1]:
                    malig_count += 1.0
            if malig_count > k/2:
                y_pred.append(y_train[1])
            else:
                y_pred.append(y_train[0])

        return y_pred

    x_train = full_data[:int(len(full_data) * .8)] #first 80% of datapoints
    x_test =  full_data[int(len(full_data) * .8):] #last 20% of datapoints
    y_train = [0.0,1.0]
    #run the k-nn algo
    y_pred = knn_classifier(x_test, x_train, y_train, int(k), 2)
    #check and print accuracy
    correct = 0
    for i in range(0, len(y_pred)):
        if x_test[i][-1] == y_pred[i]:
            correct += 1
    print("part 1 accuracy:",correct/len(y_pred))
