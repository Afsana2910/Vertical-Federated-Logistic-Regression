import numpy as np
from LR import model
from client_interface import ClientInterface

class Host(ClientInterface):
    def __init__(self,lr,model:model,data):
        self.lr = lr
        self.x = data
        self.model = model(self.x,self.lr)
        self.z= None
        self.diff = None

    def create_batch(self,ids):

        return np.array([self.x[id] for id in ids])

    def forward(self,ids):
        x=self.create_batch(ids)
        self.z = self.model.forward(x)

    def receive(self,diff):
        #self.dw,self.db=grad
        self.diff = diff
    
    def send(self):
        return self.z

    def compute_gradient(self):
         self.dw,self.db = self.model.compute_gradient(self.diff)
    
    def update_model(self):

        self.model.update_model()

    

    