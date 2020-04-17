import math
import numpy as np
import csv
import random

def load_dataset(path):
    dataset = []
    with open(path) as file:
        reader = csv.reader(file)
        for row in reader:
            dataset.append(row)
    return dataset

def str_to_float(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            dataset[i][j] = float(dataset[i][j])

def minMaxFunction(dataset):
    minMax = []
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        minVal = min(col_values)
        maxVal = max(col_values)
        minMax.append([minVal,maxVal])
    return minMax

def normalization(dataset,minmax):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            numer = dataset[i][j] - minmax[j][0]
            denom = minmax[j][1] - minmax[j][0]
            dataset[i][j] = numer/denom

def crossValidation(dataset,k=5):
    dataset_copy = list(dataset)
    size = len(dataset) // k
    folds = []
    for i in range(k):
        fold = []
        while len(fold) < size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        folds.append(fold)
    return folds
            

def accuracy_metrics(actual,pred):
    count = 0
    for i in range(len(actual)):
        if actual[i] == pred[i]:
            count += 1
    return count / len(actual) * 100

def prediction(x,w):
    z = np.dot(x,w)
    return 1 / (1 + np.exp(-z))

def cost_function(features,target,weights):
    n = len(target)
    y_pred = prediction(features,weights)
    cost_1 = -target * np.log(y_pred)
    cost_2 = -(1 - target) * np.log((1 - y_pred))
    cost = cost_1 + cost_2
    total = np.sum(cost) / n
    return total

def gradientDescent(x,y,epochs,alpha):
    w = [0] * len(x[0])
    n = len(x)
    for epoch in range(epochs):
        pred = prediction(x,w)
        loss = pred - y
        grad = (2/n) * (loss.dot(x))
        w = w - alpha * grad
        if epoch % 1000 == 0:    
            cost = cost_function(x,y,w)
            print("Cost is",cost)
    return w
    

def logisticRegression(x_train,y_train,x_test,y_test,epochs,alpha):
    weights = gradientDescent(x_train,y_train,epochs,alpha)
    y_pred = prediction(x_test,weights)
    for i in range(len(y_pred)):
        y_pred[i] = round(y_pred[i])
    score = accuracy_metrics(y_test,y_pred)
    return score
    
def evaluateAlgorithm(dataset,epochs,alpha):
    folds = crossValidation(dataset)
    scores = []
    for fold in folds:
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        train = list(folds)
        train.remove(fold)
        for train_fold in train:
            for data in train_fold:
                x_train.append(data[:-1])
                y_train.append(data[-1])
        
        for data in fold:
            x_test.append(data[:-1])
            y_test.append(data[-1])
        x_train = np.asarray(x_train)
        x_test = np.asarray(x_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        score = logisticRegression(x_train,y_train,x_test,y_test,epochs,alpha)
        print("Score is",score)
        scores.append(score)
    return scores

filename = 'data.csv'
dataset = load_dataset(filename)
str_to_float(dataset)
minMax = minMaxFunction(dataset)
normalization(dataset,minMax)

epochs = 50000
alpha = 0.01
acc = evaluateAlgorithm(dataset,epochs,alpha)