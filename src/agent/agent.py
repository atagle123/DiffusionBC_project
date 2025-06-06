from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

    ### inference ###
    @abstractmethod
    def config_policy(self):
        pass

    @abstractmethod
    def policy(self):
        pass
