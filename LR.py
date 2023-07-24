from typing import overload
import numpy as np
from multipledispatch import dispatch
import scipy as sp 
sp.random.seed(12345) 


class model:

    def __init__(self,data,lr):
        self.lr = lr
        x=data
        self.m,self.n=x.shape
        self.w=sp.random.normal(loc=0.0, scale=1.0, size=(self.n,1))
        self.b=0
        self.lr=lr
        self.dw=None
        self.db=None

    def normalize(self,X):
        X = (X - X.mean(axis=0))/X.std(axis=0)
        return X

    def sigmoid(self,z):
        return 1.0/(1 + np.exp(-z))
    
    def loss(self,y, y_hat):
        loss = -np.mean(y*(np.log(y_hat)) + (1-y)*np.log(1-y_hat))
        return loss
    
    def gradients(self,X, diff):
        # Gradient of loss w.r.t weights.
        dw = (1/self.m)*np.dot(X.T, diff)
        # Gradient of loss w.r.t bias.
        db = (1/self.m)*np.sum(diff) 
        return dw, db
    
    def forward(self,x):
        #self.x = self.normalize(x)
        self.x = x
        z = (np.dot(self.x, self.w) + self.b)
        return z

    def compute_diff(self,z,y):
        y_hat=self.sigmoid(z)
        diff = y_hat - y
        return diff
    
    def compute_gradient(self,diff):
        self.dw, self.db = self.gradients(self.x, diff)
        return self.dw,self.db


    def update_model_(self,y):
        #self.dw=dw
        #self.db=db
        # Updating the parameters.
        self.w -= self.lr*self.dw
        self.b -= self.lr*self.db
        l = self.loss(y, self.sigmoid(np.dot(self.x, self.w) + self.b))
        return l
    def update_model(self):
        #self.dw=dw
        #self.db=db
        # Updating the parameters.
        self.w -= self.lr*self.dw
        self.b -= self.lr*self.db
        


    def get_gradients(self):
        if (self.dw and self.db) is not None:
            return self.dw, self.db
        else:
            return None

    def predict(self, X):
        #x = self.normalize(X)
        self.x = X
        preds = self.sigmoid(np.dot(self.x, self.w) + self.b)
        pred_class = []
        pred_class = [1 if i > 0.5 else 0 for i in preds]
        return np.array(pred_class)


    def accuracy(self, y, y_hat):
        y = y.reshape(-1,1)
        y_hat = y_hat.reshape(-1,1)
        accuracy = np.sum(y == y_hat) / len(y)
        return accuracy
