
class ProcessPointer:
    def __init__(self, process_list):
        self.process_list = process_list
        self.names = []
        self.callbacks = {}
        self.position = 0
    
    def sync(self):
        self.names = list(self.process_list.keys())
    
    def get_node(self):
        return self.process_list[self.names[self.position]]

    def get_index(self, name):
        return self.names.index(name)

    def move(self, index):
        self.position += index
        assert  0 <= self.position < len(self.process_list)

        removing_callback = None
        if self.position in self.callbacks:
            for callback in reversed(self.callbacks[self.position]):
                if not callback():
                    removing_callback = callback
                    break

        if removing_callback is not None:
            callbacks = self.callbacks[self.position]
            del self.callbacks[self.position][callbacks.index(removing_callback)]


    def add_callback(self, index, callback):
        """Add callback function when to move index.

        Args:
            index (int, str): index or name corresponding callback.
            callback (function): callback function. If return value is False, 
                                 this loop will be ended.
        """

        if isinstance(index, str):
            index = self.names.index(index)
        self.callbacks.setdefault(index, []).append(callback)

    def forward(self, start = None):
        if start is not None:
            self.position = start

        while self.position < len(self.process_list):
            yield self.get_node()
            self.position = self.position + 1

    def backward(self, start = None):
        if start is not None:
            self.position = start

        while self.position >= 0:
            yield self.get_node()
            self.position = self.position - 1

    def __iadd__(self, index):
        self.move(index)

    def __isub__(self, index):
        self.move(-index)