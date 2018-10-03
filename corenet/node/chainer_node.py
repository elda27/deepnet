from .updatable_node import UpdatableNode
from chainer import cuda


class ChainerNode(UpdatableNode):
    Updaters = {}

    def update_before(self):
        self.model.cleargrads()

    def update_core(self, loss):
        xp = cuda.get_array_module(loss)
        if xp.isnan(loss.data):
            raise ValueError('Loss is NaN: {}'.format(self.update_variable))
        loss.backward()

    def update_after(self):
        ChainerNode.Updaters[self.update_variable].update()

    @classmethod
    def add_updater(cls, name, updater):
        cls.Updaters[name] = updater
