from typing import Tuple, List
import numpy as np
from model import LogisticRegressionModel
from client_interface import ClientInterface

class Guest(ClientInterface):
    """
    Implementation of the active party (Guest) in a federated learning system.
    """
    def __init__(self, lr: float, model_instance: LogisticRegressionModel, data: Tuple[np.ndarray, np.ndarray]):
        """
        Initializes the Guest instance.

        Parameters:
        lr (float): Learning rate.
        model_instance (model): Instance of the logistic regression model.
        data (Tuple[np.ndarray, np.ndarray]): Tuple containing features and labels.
        """
        self.lr = lr
        self.x, self.y = data
        self.y = self.y.reset_index(drop=True) 
        self.model = model_instance(self.x, self.lr)
        self.z = None
        self.diff = None

    def create_batch(self, ids: List[int]) -> np.ndarray:
        """Creates a batch of data using the provided ids."""
        self.y_ = np.array([self.y[id] for id in ids]).reshape(-1, 1)
        return np.array([self.x[id] for id in ids])

    def forward(self, ids: List[int]):
        """Performs a forward pass using the data associated with the provided ids."""
        x = self.create_batch(ids)
        
        self.z = self.model.forward(x)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalizes the input data."""
        return (X - X.mean(axis=0)) / X.std(axis=0)

    def receive(self, z1: np.ndarray, z2: np.ndarray):
        """Receives data from passive parties and updates z."""
        self.z += z1 + z2

    def compute_gradient(self):
        """Computes the gradient based on received data and local parameters."""
       
        self.diff = self.model.compute_diff(self.z, self.y_)
        self.dw, self.db = self.model.compute_gradient(self.diff)

    def send(self) -> np.ndarray:
        """Sends computed diff to other parties."""
        return self.diff

    def update_model(self):
        """Updates the model parameters and computes the loss."""
        self.loss = self.model.update_model_(self.y_)

    def predict(self, X_guest, host_contributions):
        # X_guest is the input features available to the Guest
        z_guest = np.dot(X_guest, self.model.w) + self.model.b
        # Sum up contributions from all hosts
        z_total = z_guest + sum(host_contributions)
        prob = self.model.sigmoid(z_total)
        predictions = [1 if p >= 0.5 else 0 for p in prob]
        return predictions

    def predict_local(self, X_guest):
        # X_guest is the input features available to the Guest
        z_guest = np.dot(X_guest, self.model.w) + self.model.b
        prob = self.model.sigmoid(z_guest)
        predictions = [1 if p >= 0.5 else 0 for p in prob]
        return predictions   
        
