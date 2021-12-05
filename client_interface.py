from abc import ABC, abstractmethod

class ClientInterface(ABC):
    @abstractmethod
    def receive(self, data):
        ...
    @abstractmethod
    def send(self):
        ...
    @abstractmethod
    def compute_gradient(self):
        ...
    