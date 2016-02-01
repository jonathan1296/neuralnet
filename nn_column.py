# coding: utf-8

'''
This code uses the NeuralNetwork class to predict column abnormalities.

Author: Jonathan Arriaga
Jan 30, 2016
'''

import sys

from scipy.optimize import minimize
import numpy as np
import pandas as pd

sys.path.append( 'C:/Users/jonathan/Documents/python' )

import modules.neuralnet as nn

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


# ~~~~~ New NerualNetwork object to train ~~~~~
nncol = nn.NeuralNet()

nncol.inputs = 1
nncol.outputs = 3
nncol.layer_size = [3,3] # Hidden layers


column_data = pd.read_csv('data/column_2C.dat', sep='\s+', header=None, 
                          skiprows=1)

X = np.array(column_data[[0,1,2,3,4,5]])



Y = np.sin(X)

nn.initializeThetas()
nn.setTrainingData(X, Y, epochs=50000, output_type='linear', learning_rate=0.1, 
                   disp=50, reg_lambda=1)

#nn.predict(X)
J_train = nn.train()

cFigure = 0

cFigure+=1
figure(cFigure)
plot(J_train[1:]) # Don't plot first iteration, errors are high
ylabel("Cost")
xlabel("Epochs")

cFigure+=1
figure(cFigure)
plot(np.log(J_train))
ylabel("log(Cost)")
xlabel("Epochs")

cFigure+=1
figure(cFigure)
Ypred = nn.predict(X)
plot(X,Y)
plot(X,Ypred)
ylabel("Y = sin(X)")
xlabel("X")



# ~~~~~ Using scipy minimizers witn Y=sin(X) examples
nn = NeuralNetwork()
nn.inputs = 1
nn.outputs = 1
nn.layer_size = [5,5] # Hidden layers

X = np.linspace(-4,4,1000)
Y = np.sin(X)

nn.initializeThetas()
nn.setTrainingData(X, Y, epochs=50000, output_type='linear', learning_rate=0.1, 
                   disp=50, reg_lambda=1)

# Minimize thetas using minimizer
Thetas_ur = nn.unrollThetas()
res = minimize(nn, Thetas_ur, method='BFGS', jac=True, 
                  options={'disp': True, 'maxIter':1000})
Thetas_ur_opt = res['x']
nn.reshapeThetas(Thetas_ur_opt)

cFigure+=1
figure(cFigure)
Ypred = nn.predict(X)
plot(X,Y)
plot(X,Ypred)
ylabel("Y = sin(X)")
xlabel("X")


# ~~~~~ Logistic regression
ex4data = pd.read_csv('/home/jonathan/python/test/ex4data1.csv', 
                      names=[str('f'+str(i)) for i in range(400)]+[str('y')])
ex4data = np.array(ex4data)
np.random.shuffle(ex4data)
cols = 400
X = np.array(ex4data[:,:cols])
y = np.array(ex4data[:,-1])


# Reduce training examples
X = X[0:int(0.1*X.shape[0]), ]
y = y[0:int(0.1*y.size)]

# Using PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=0.9)
pca.fit(X)
X_red = pca.transform(X)
cols = X_red.shape[1]

ybin = np.zeros((len(X), 10))
for i in range(len(y)): 
    if y[i] == 10: y[i] = 0
    ybin[i,int(y[i])] = 1

nn = NeuralNetwork()
nn.inputs = cols
nn.outputs = 10
nn.layer_size = [25] # Hidden layers

nn.initializeThetas()
nn.setTrainingData(X_red, ybin, epochs=1000, output_type='logistic', learning_rate=0.3, 
                   disp=50)

Thetas_ur = nn.unrollThetas()
Thetas = minimize(nn, Thetas_ur, method='BFGS', jac=True, 
                  options={'gtol': 0.5})


J_train = nn.train()

cFigure+=1
figure(cFigure)
plot(J_train[1:]) # Don't plot first iteration, errors are high
ylabel("Cost")
xlabel("Epochs")

cFigure+=1
figure(cFigure)
plot(np.log(J_train))
ylabel("log(Cost)")
xlabel("Epochs")


# Using PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=0.9)
pca.fit(X)
X_red = pca.transform(X)
cols = X_red.shape[1]

nn = NeuralNetwork()
nn.inputs = cols
nn.outputs = 10
nn.layer_size = [25] # Hidden layers

nn.initializeThetas()
nn.setTrainingData(X_red, ybin, epochs=1000, output_type='logistic', learning_rate=0.3, 
                   disp=50)

J_train = nn.train()

cFigure+=1
figure(cFigure)
plot(J_train[1:]) # Don't plot first iteration, errors are high
ylabel("Cost")
xlabel("Epochs")

cFigure+=1
figure(cFigure)
plot(np.log(J_train[1:]))
ylabel("log(Cost)")
xlabel("Epochs")

cFigure+=1
figure(cFigure)
plot(nn.accuracy_hist) # Don't plot first iteration, errors are high
ylabel("Accuracy")
xlabel("Epochs")





import numpy as np
from scipy.optimize import minimize

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

def rosen(x):
    """The Rosenbrock function"""
    return [sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)]


x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

res = minimize(rosen, x0, jac=rosen_der, method='BFGS', 
               options={'disp': True})


def rosen_cost_der(x):
    """The Rosenbrock function"""
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    J = sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return J, der

res = minimize(rosen_cost_der, x0, jac=True, method='BFGS', 
               options={'disp': True})




