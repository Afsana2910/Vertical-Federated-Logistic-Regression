from abc import ABC, abstractmethod

class ClientInterface(ABC):


    @abstractmethod
    def receive(self, data) -> None:
        pass
    
    @abstractmethod
    def send(self) -> None:
       
        pass
    
    @abstractmethod
    def compute_gradient(self) -> None:
       
        pass
