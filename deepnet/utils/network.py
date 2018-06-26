from functools import partial
from time import sleep
import concurrent.futures
import contextlib 
import test
import enum

class NetworkNode:
    def __init__(self, input, output, model, training=True, validation=True, test=True, updatable = False, **args):
        input_ = input
        self.input = input_ if isinstance(input_, list) else [ input_ ] 
        self.output = output if isinstance(output, list) else [ output ] 
        
        assert all([isinstance(var, str) for var in self.input]) , 'Input must be string: {}'.format(self.input)
        assert all([isinstance(var, str) for var in self.output]), 'Output must be string: {}'.format(self.output)

        self.model = model
        self.updatable = updatable
        self.training = training
        self.validation = validation
        self.test = test

        self.args = args
        self.callback = None
        self.clear_state()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def update(self):
        if not self.updatable:
            return
        self.model.cleargrads()

    def ready(self, name):
        self.is_already_[name] = True
        if self.is_already() and self.callback is not None:
            self.callback()

    def is_already(self):
        return all(self.is_already_.values())
    
    def set_callback(self, callback):
        self.callback = callback

    def clear_state(self, clear_callback = True):
        if clear_callback:
            self.callback = None
        self.is_already_ = { i: False for i in self.input }

class NetworkWalker:
    def __init__(self, network, input_list):
        self.network = network
        self.clear_network_state(network)

        # Search root nodes
        self.input_list = input_list
        self.start_nodes = []
        self.node_dependency = {}
        self.walk_nodes()

    def walk_nodes(self):
        for node in self.network.values():
            for var in node.input:
                self.node_dependency.setdefault(var, []).append(node)
                if var in self.input_list:
                    node.ready(var)
            if node.is_already():
                self.start_nodes.append(self.network)

        self.clear_network_state(self.network)

    def clear_network_state(self, clear_callback = True):
        for node in self.network.values():
            node.clear_state(clear_callback)

class InferenceMode(enum.Enum):
    Train = 1
    Validation = 2
    Test = 3

class NetworkManager:
    def __init__(self, input_list):
        self.network = {}
        self.updatable_node = []
        self.input_list = input_list
        self.variables = {}
        self.futures = []
        self.mode_list = {
            'train': InferenceMode.Train, 
            'valid': InferenceMode.Validation, 
            'test':  InferenceMode.Test, 
        }
        
    def add(self, node_name, node):
        assert node_name not in self.network
        self.network[node_name] = node
        if node.updatable:
            self.updatable_node.append(node)

    def start_process(self, node):
        if self.mode == InferenceMode.Validation and not node.validation: # If not running on validation  
            return
        elif self.mode == InferenceMode.Train and not node.training: # If not running on training
            return
        elif self.mode == InferenceMode.Test and not node.test: # If not running on test
            return

        in_values = [ self.variables[var] for var in node.input ]
        out = node(*in_values, **node.args)
        if not isinstance(out, (list, tuple)):
            out = [ out ]

        for name, var in zip(node.output, out):
            #assert name in self.walker.node_dependency, \
            #  'Unkwown input label: {}\nDependency node input are following: {}'.format(
            #      name, 
            #      ', '.join(self.walker.node_dependency.keys())
            #      )
                
            self.variables[name] = var
            if name not in self.walker.node_dependency:
                continue

            for dep_node in self.walker.node_dependency[name]:
                dep_node.ready(name)

    def update(self):
        for node in self.updatable_node:
            node.update()

    def __call__(self, mode='train', **inputs):
        self.variables = {}
        self.variables.update(inputs)
        self.walker = NetworkWalker(self.network, self.input_list)
        self.walker.clear_network_state()
        self.mode = self.mode_list[mode]

        for node in self.network.values():
            node.set_callback(partial(NetworkManager.start_process, self, node))

        for name in inputs.keys():
            if name not in self.walker.node_dependency:
                continue
            for node in self.walker.node_dependency[name]:
                node.ready(name)
    
