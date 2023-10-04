from typing import List, Tuple
import numpy as np
from model import LogisticRegressionModel
from client_interface import ClientInterface

class Host(ClientInterface):
    """
    Implementation of a passive party (Host) in a vertical federated learning system.
    """
    def __init__(self, lr: float, model_instance: LogisticRegressionModel, data: np.ndarray):
        """
        Initializes the Host instance.

        Parameters:
        lr (float): Learning rate.
        model_instance (model): Instance of the logistic regression model.
        data (np.ndarray): Data for the host.
        """
        self.lr = lr
        self.x = data
        self.model = model_instance(self.x, self.lr)
        self.z = None
        self.diff = None

    def create_batch(self, ids: List[int]) -> np.ndarray:
        """Creates a batch of data using the provided ids."""
        return np.array([self.x[id] for id in ids])

    def forward(self, ids: List[int]):
        """Performs a forward pass using the data associated with the provided ids."""
        x = self.create_batch(ids)
        self.z = self.model.forward(x)

    def receive(self, diff: np.ndarray):
        """Receives diff from the active party (Guest)."""
        self.diff = diff

    def send(self) -> np.ndarray:
        """Sends computed z to the active party (Guest)."""
        return self.z

    def compute_gradient(self):
        """Computes the gradient based on received diff and local parameters."""
        self.dw, self.db = self.model.compute_gradient(self.diff)

    def update_model(self):
        """Updates the model parameters."""
        self.model.update_model()

    def compute_contribution(self, X_host):
            # X_host is the input features available to the Host
            return np.dot(X_host, self.model.w) + self.model.b
