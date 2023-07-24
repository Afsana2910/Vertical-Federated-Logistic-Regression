import numpy as np
from LR import model
from client_interface import ClientInterface

class Guest(ClientInterface):
    def __init__(self,lr,model:model,data):
        self.lr = lr
        self.x,self.y = data
        self.model = model(self.x,self.lr)
        self.z = None
        self.diff = None
        
    
    def create_batch(self,ids):
        self.y_=np.array([self.y[id] for id in ids])
        self.y_=self.y_.reshape(len(self.y_),1)
        return np.array([self.x[id] for id in ids])

    def forward(self,ids):
        x=self.create_batch(ids)
        self.z = self.model.forward(x)

    def normalize(self,X):
        X = (X - X.mean(axis=0))/X.std(axis=0)
        return X

    def receive(self,z1,z2):


        #self.z = self.normalize(self.z)
        self.z = self.z + z1 + z2
       
    def compute_gradient(self):
        self.diff = self.model.compute_diff(self.z,self.y_)
        self.dw,self.db = self.model.compute_gradient(self.diff)
    
    def send(self):
        #return self.dw, self.db
        return self.diff
    
    def update_model(self):
        self.loss = self.model.update_model_(self.y_)

    

    
