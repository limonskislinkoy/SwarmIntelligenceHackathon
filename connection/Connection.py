from abc import ABC, abstractmethod


class Connection(ABC):
    @abstractmethod
    def set_connection(self):
        pass

    @abstractmethod
    def receive_data(self):
        pass

    @abstractmethod
    def send_data(self, data):
        pass

    @abstractmethod
    def close_connection(self):
        pass
