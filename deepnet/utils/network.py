from functools import partial
from time import sleep
import concurrent.futures
import contextlib 
import test
import enum
from collections import OrderedDict
import networkx as nx

class NetworkNode:
    def __init__(self, 
        name, 
        input, output, model, 
        training=True, validation=True, test=True, 
        updatable = False, iterate_from=None,
        args=dict()
        ):
        input_ = input
        self.name  = name
        self.input = input_ if isinstance(input_, list) else [ input_ ] 
        self.output = output if isinstance(output, list) else [ output ] 
        self.iterate_from_node = iterate_from
        
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
        return self.iterate_from_node is not None:

    def set_callback(self, callback):
        self.callback = callback

    def clear_state(self, clear_callback = True):
        if clear_callback:
            self.callback = None
        self.is_already_ = { i: False for i in self.input }

def clear_network_state(network, clear_callback = True):
        for node in network.nodes(data='node').values():
            node.clear_state(clear_callback)

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

class NetworkWalker_:
    def __init__(self, network):
        self.network = network
        clear_network_state(self.network)

    def walk_impl(self, node):
        next_nodes = []
        iteration_edges = []
        for next_node_name in g[node.name]:
            next_node = g.nodes[next_node_name]['node']
            edge = self.network.edges[node.name, next_node_name]
            if 'input' in edge:  # If true, this egde is the flow of egde.
                next_node.ready(edge['input'])
                if next_node.is_already():
                    next_nodes.append(next_node)
            elif 'iteration' in edge:
                iteration_edges.append(edge)

        for edge in iteration_edges:
            # :TODO: Append condition check for looping process.
            raise NotImplementedError()

        return next_nodes

class NetworkBuilder:
    def __init__(self, graph):
        self.graph = graph
        self.start_nodes = []

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
            self.already_node_names = [ node.name for node in already_nodes ]
            self.start_nodes = already_nodes
            
        for node in self.already_nodes:
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
            self.variables[name] = var
            if name not in self.walker.node_dependency:
                continue

            for dep_node in self.walker.node_dependency[name]:
                dep_node.ready(name)

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
        # self.walker = NetworkWalker(self.network, self.input_list)
        # self.walker.clear_network_state()
        self.mode = self.mode_list[mode]

        # for node in self.network.values():
        #     node.set_callback(partial(NetworkManager.start_process, self, node))

        # for name in inputs.keys():
        #     if name not in self.walker.node_dependency:
        #         continue
        #     for node in self.walker.node_dependency[name]:
        #         node.ready(name)

        ravel_nodes = []
        for node in self.start_nodes:
            self.network[node.name]
