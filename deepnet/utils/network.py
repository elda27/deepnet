from functools import partial
from time import sleep
import concurrent.futures
import contextlib 
import test
import enum
from collections import OrderedDict
import networkx as nx
import abc
import types

class NetworkNode:
    def __init__(self, 
        name, 
        input, output, model, 
        training=True, validation=True, test=True, 
        updatable = False, iterate_from = None,
        args=dict()
        ):
        input_ = input
        self.name  = name
        self.input = input_ if isinstance(input_, list) else [ input_ ] 
        self.output = output if isinstance(output, list) else [ output ] 
        self.iteration_from_node = iterate_from
        
        assert all([isinstance(var, str) for var in self.input]) , 'Input must be string: {}'.format(self.input)
        assert all([isinstance(var, str) for var in self.output]), 'Output must be string: {}'.format(self.output)

        self.model = model
        if isinstance(updatable, bool):
            assert not updatable, "updatable must be False or variable name."
            self.update_variable = None
        else:
            self.update_variable = updatable

        self.training = training
        self.validation = validation
        self.test = test

        self.args = args
        self.callback = None
        self.clear_state()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __repr__(self):
        return str(dict(input=self.input, output=self.output))

    @property
    def updatable():
        return self.update_variable is not None

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
    
    def is_iterable(self):
        return self.iteration_from_node is not None

    def set_callback(self, callback):
        self.callback = callback

    def clear_state(self, clear_callback = True):
        if clear_callback:
            self.callback = None
        self.is_already_ = { i: False for i in self.input }

def clear_network_state(network, clear_callback = True):
        for node in network.nodes(data='node').values():
            node.clear_state(clear_callback)

class ControlNode:
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, network, ):
        raise NotImplementedError()

class IteratableProcessor:
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def insert(self, *args):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_output(self):
        raise NotImplementedError()

class IteratingNode(NetworkNode, ControlNode):
    def __init__(self, start_node, walker):
        NetworkNode.__init__(
            None, start_node.input, start_node.output, start_node.model,
            training=start_node.training,
            validation=start_node.validation,
            test=start_node.test,
            args=start_node.args
        )
        self.start_node = start_node
        self.walker = walker

    def __call__(self, *input, **args):
        path = nx.all_simple_paths(self.walker.network, source=self.start_node.name, target=self.start_node_name.iteration_from_node)[1:]
        output_node = self.walker.network[self.start_node.iterate_from]['node']
        for input_ in self.start_node:
            self.walker.variables.update(dict(zip(self.start_node.input, input_)))
            for node in path:
                output = self.walker.invoke(node)
                self.walker.variables.update(dict(zip(node.output, output)))
            output = [ self.walker.variables[out] for out in output_node.output ]
            self.start_node.model.insert(*output)
        return self.start_node.model.get_output()

class NetworkWalker:
    def __init__(self, network, mode, variables):
        self.network = network
        self.mode = mode
        self.iteration_stack = []
        self.variables = variables
        clear_network_state(self.network)

    def update_variables(self, node, values):
        output = { out: value for out, value in zip(node.output, values) }
        self.variables.update(**output)


    def start(self, start_nodes):
        for node in start_nodes:
            output = self.invoke(node)
            self.update_variables(node, output)
        
        next_node_names = [ node.name for node in start_nodes ]
        while len(next_node_names) == 0:
            next_nodes = self.walk(next_node_names)
            next_node_names = [ node.name for node in next_nodes ]

    def walk(self, start_node_names):
        already_nodes = []
        for node_name in start_node_names:
            node = self.network.nodes[node_name]['node']
            if node.is_iterable(): # Iterable nodes found.
                already_nodes.append(IteratingNode(node, self))
            else:
                already_nodes.extend(self.search_next_node(node))

        for node in already_nodes:
            output = self.invoke(node)
            self.update_variables(node, output)

        return already_nodes

    def search_next_node(self, node):
        already_nodes = []
        for next_node_name in self.network[node.name]: # Iterate about children nodes.
            edge = self.network.edges[node.name, next_node_name]
            if 'input' in edge:  # If true, this egde is the flow of egde.
                next_node = self.network.nodes[next_node_name]['node']
                next_node.ready(edge['input'])
                if next_node.is_already():
                    already_nodes.append(next_node)
        return already_nodes

    def invoke(self, node, variables = None):
        if variables is None:
            variables = self.variables
        if self.mode == InferenceMode.Validation and not node.validation: # If not running on validation  
            return
        elif self.mode == InferenceMode.Train and not node.training: # If not running on training
            return
        elif self.mode == InferenceMode.Test and not node.test: # If not running on test
            return

        in_values = [ variables[var] for var in node.input ]
        out = node(*in_values, **node.args)
        if isinstance(out, types.GeneratorType):
            assert node.is_iterable(), "This node is not iterable but using process is corutine or generator function."
        elif not isinstance(out, (list, tuple)):
            out = [ out ]

        return out

class NetworkBuilder:
    def __init__(self, graph):
        self.graph = graph
        self.start_nodes = []
        self.already_node_names = []

    def build(self, input_list, source_node = None):
        already_nodes = []
        
        clear_network_state(self.graph)
        for node in self.graph.nodes(data='node').values():
            if node.name in self.already_node_names:
                continue

            for input_ in input_list:
                if input_ in node.input:
                    node.ready(input_)
                    self.graph.add_edge(source_node, node.name, input=input_)

            if node.is_iterable(): # If iterable node, setup to loop edge
                self.graph.add_edge(node.from_node, source_node, iteration=True)

            if node.is_already():
                already_nodes.append(node)

        if source_node is None:
            self.already_node_names.extend( [ node.name for node in already_nodes ] )
            self.start_nodes = already_nodes
            
        for node in already_nodes:
            self.build(node.output, source_node = node.name)

class InferenceMode(enum.Enum):
    Train = 1
    Validation = 2
    Test = 3

class NetworkManager:
    def __init__(self, input_list):
        self.network = nx.DiGraph()
        self.updatable_node = []
        self.input_list = input_list
        self.variables = {}
        self.futures = []
        self.mode_list = {
            'train': InferenceMode.Train, 
            'valid': InferenceMode.Validation, 
            'test':  InferenceMode.Test, 
        }
        
    def add(self, node):
        assert node.name not in self.network,\
            'Duplicating label name: {}({})<{}>'.format(node.name, node, str({ key: str(node) for key, node in self.network.items() }))
        self.network.add_node(node.name, node=node)
        if node.updatable:
            self.updatable_node.append(node)

    def build_network(self):
        builder = NetworkBuilder(self.network)
        builder.build(self.input_list)
        self.start_nodes = builder.start_nodes

    def validate_network(self, not_reached):
        # Search not found node.
        def search_node(name):
            found_node = None
            for node in self.network.values(): 
                if name in node.output:
                    found_node = node
                    break
            return found_node

        not_reached_node = search_node(not_reached)

        if not_reached_node is None:
            return []

        found_nodes = []

        for name, is_ready in not_reached_node.is_already_.items():
            if not is_ready:
                found_nodes.append(search_node(name))
                found_nodes.extend(self.validate_network(name))

        return found_nodes

    def update(self):
        for node in self.updatable_node:
            node.update()

    def __call__(self, mode='train', **inputs):
        assert all((name in inputs for name in self.input_list)) or mode == 'test', \
            'Input requirement is not satisfied. (Inputs: {}, Input requirement: {}])'.format(list(inputs.keys()), self.input_list)
        #assert len(self.network.edges) != 0, "Network is not initialized. you need to invoke build network."
        if len(self.network.edges) == 0:
            self.build_network()

        self.variables = {}
        self.variables.update(inputs)
        self.mode = self.mode_list[mode]

        walker = NetworkWalker(self.network, self.mode, self.variables)
        for node in self.start_nodes:
            output = walker.invoke(node)
            { node.output }

