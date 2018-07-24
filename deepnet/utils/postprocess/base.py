import abc
import enum

class PostProcessTrigger(enum.IntEnum):
    AfterEachProcess = 1 # Post process after each process.
    AfterProcess     = 2 # Post process after sequence (e.g. after validation).

class PostProcessor(metaclasss=abc.ABCMeta):
    def __init__(self, type):
        self.type = type

    @abc.abstractmethod
    def update(self, variable):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_result(self):
        raise NotImplementedError()

