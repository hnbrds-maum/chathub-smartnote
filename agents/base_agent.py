from abc import ABCMeta, abstractmethod

class BaseAgent(metaclass=ABCMeta):
    def __init__(self):
        self.build_workflow()        

    @abstractmethod
    def build_workflow(self):
        return