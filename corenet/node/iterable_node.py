from .network_node import NetworkNode
from abc import abstractmethod

class IterableNode(NetworkNode):
    def __init__(
        self, name, model,
        input, output,
        args, end_node, **kwargs
    ):
        super().__init__(name, model, input, output, args, **kwargs)
        self.distance = self.pointer.get_index(end_node) - self.get_index(name)
        self.pointer.add_callback(end_node, self.check)
        self.next_value = None

        self.slice_values = []

    def __call__(self, *args):
        self.iterator = iter(self.model(*args, **self.kwargs))
        if self.next_value is None:
            self.slice_values = []
            self.next_value = next(self.iterator)
        value = self.next_value
        try:
            self.next_value = next(iterator)
        except StopIteration:
            self.next_value = None
        
    def check(self, *values):
        if self.next_value is not None:
            self.move(-self.distance)
            for dst, src in zip(self.slice_values, values):
                dst.append(src)
            return True
        else:
            return False

    def get_values(self):
        return self.slice_values

