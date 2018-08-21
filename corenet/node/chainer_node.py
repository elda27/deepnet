from .updatable_node import UpdatableNode

class ChainerNode(UpdatableNode):
    Updaters = {}

    def update_before(self):
        self.model.cleargrads()

    def update_core(self, value):
        value.backward()
    
    def update_after(self):
        ChainerNode.Updaters[self.update].update()

    @classmethod
    def add_updater(cls, name, updater):
        cls.Updaters[name] = updater