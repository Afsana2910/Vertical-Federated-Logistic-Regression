import numpy as np
from LR import model
from client_interface import ClientInterface

class Guest(ClientInterface):
    def __init__(self,lr,model:model,data):
        self.lr = lr
        self.x,self.y = data
        self.model = model(self.x,self.lr)
        self.z = None
        
    
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
        #_z = self.normalize(_z)
        #self.z = (_z + self.z) / 2
        #self.z = _z + self.z
        #self.z = np.mean(_z,self.z)
        self.z = self.z + z1 + z2
        #self.z = (0.25*self.z) + (0.5*z1) + (0.25*z2)
        #self.z = ((0.7*self.z) + (0.15*z1) + (0.15*z2))/3
        #self.z = (self.z + z1 + z2)/3
    
    def compute_gradient(self):
        self.dw,self.db = self.model.compute_gradient(self.z,self.y_)
    
    def send(self):
        return self.dw,self.db
    
    def update_model(self):
        self.loss = self.model.update_model_(self.dw,self.db,self.y_)

    

    