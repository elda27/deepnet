from .network_node import NetworkNode
from contextlib import contextmanager
from abc import abstractmethod


class UpdatableNode(NetworkNode):
    def __init__(self, update=None, **kwargs):
        super().__init__(**kwargs)
        self.update_variable = update

    def update(self, variable):
        yield self.update_before()
        yield self.update_core(variable[self.update_variable])
        yield self.update_after()

    def update_before(self):
        pass

    def update_core(self, value):
        pass

    def update_after(self):
        pass
