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
from datetime import datetime
import hashlib
import itertools
from copy import deepcopy
import chainer
import chainer.functions as F
from deepnet.core import config 
from deepnet.utils import functions as utils
from logging import getLogger, DEBUG

logger = getLogger(__name__)

def get_unique_label():
    sleep(1e-6)
    now = str(datetime.now()).encode('ascii')
    return hashlib.md5(now).hexdigest()

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
    def updatable(self):
        return self.update_variable is not None

    def update(self):
        if not self.updatable:
            return
        self.model.cleargrads()

    def ready(self, name):
        if name not in self.is_already_:
            return

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

    def get_param(self, fields):
        proc = utils.get_field(self.model, fields[:-1])
        return proc.stores[fields[-1]]


class ParallelNetworkNode(NetworkNode):
    """GPU parallelization for chainer network.

    I/O data is assume zero of gpu id.
    """

    def __init__(self, 
        name, 
        input, output, model, 
        training=True, validation=True, test=True, 
        updatable = False, iterate_from = None,
        args=dict()
        ):
        super().__init__(
            name, input, output, model, 
            training=training, validation=validation, test=test, 
            updatable=updatable, iterate_from=iterate_from,
            args=args,
            )
        self.parallel_models = [ self.model ]
        self.gpu_ids = config.get_global_config('gpu_id')
        self.batch_size = config.get_global_config('batch_size')
        self.is_serial = False
        if not isinstance(self.gpu_ids, (tuple, list)):
            self.gpu_ids = [ self.gpu_ids ]
            self.is_serial = True
            return
        
        assert 0 in self.gpu_ids, "Main GPU assume gpu id 0."

        self.gpu_ids = list(sorted(self.gpu_ids))
        for gpu_id in self.gpu_ids:
            if gpu_id == 0:
                continue
            
            par_model = deepcopy(self.model)
            par_model.to_gpu(gpu_id)
            self.parallel_models.append(par_model)
    
    def __call__(self, *args, **kwargs):
        if self.is_serial:
            return super().__call__(*args, **kwargs)

        for model in self.parallel_models[1:]: # Update last trained parameter
            model.copyparams(self.model)

        results = []
        offset = 0
        for gpu_id, batch_size, model in zip(self.gpu_ids, self.batch_size, self.parallel_models):
            gpu_args, gpu_kwargs = self.copy_gpu(gpu_id, slice(offset, offset + batch_size), *args, **kwargs)
            result = model(*gpu_args, **gpu_kwargs)
            if gpu_id == 0:
                results.append(result)
            else:
                results.append(F.copy(result, 0))
            offset += batch_size
        
        return F.concat(results, axis=0)

    def copy_gpu(self, gpu_id, index_slice, *args, **kwargs):
        gpu_args = []
        gpu_kwargs = {}
        
        for arg in args:
            if isinstance(arg, chainer.Variable):
                tmp = F.copy(arg[index_slice], gpu_id)
                if tmp.ndim != arg.ndim:
                    tmp = F.expand_dims(tmp, axis=0)
                gpu_args.append(tmp)
            else:
                gpu_args.append(arg)
        
        for key, arg in kwargs.items():
            if isinstance(arg, chainer.Variable):
                tmp = F.copy(arg[index_slice], gpu_id)
                if tmp.ndim != arg.ndim:
                    tmp = F.expand_dims(tmp, axis=0)
                gpu_kwargs[key] = tmp
            else:
                gpu_kwargs[key] = arg
                
        return gpu_args, gpu_kwargs

    def get_param(self, fields):
        if self.is_serial:
            return super().get_param(fields)

        results = []
        for gpu_id, model in zip(self.gpu_ids, self.parallel_models):
            proc = utils.get_field(model, fields[:-1])
            results.append(F.copy(proc.stores[fields[-1]], 0))
        return F.concat(results, axis=0)

    def update(self):
        for model in self.parallel_models:
            model.cleargrads()

def clear_network_state(network, clear_callback = True):
        for _, node in network.nodes(data='node'):
            if node is None:
                continue
            node.clear_state(clear_callback)

class IterableProcessor:
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def next(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def insert(self, *args):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_output(self):
        raise NotImplementedError()

class IteratingNodeWrapper(NetworkNode):
    def __init__(self, start_node, wrap_node, network):
        assert issubclass(type(start_node.model), IterableProcessor)
        self.start_node = start_node
        self.wrap_node = wrap_node
        self.network = network
        super().__init__(
            wrap_node.name, 
            wrap_node.input, 
            wrap_node.output, 
            wrap_node.model,
            training=wrap_node.training,
            validation=wrap_node.validation,
            test=wrap_node.test,
            args=wrap_node.args
        )

    def __call__(self, *args, **kwargs):
        out = self.wrap_node(*args, **kwargs)
        if not isinstance(out, (list, tuple)):
            out = [ out ]
        self.start_node.model.insert(*out)
        try:
            self.start_node.model.next()
        except StopIteration:
            nx.set_edge_attributes(
                self.network, 
                { (self.wrap_node.name, self.start_node.name) : False}, 
                name='iteration'
                )
        return out

class NetworkWalker:
    def __init__(self, network, mode, variables):
        self.network = network
        self.mode = mode
        self.iteration_stack = []
        self.variables = variables

    def update_variables(self, node, values):
        if values is None:  # If True, invoked process don't need current inference mode.
            return
        assert isinstance(values, (tuple, list)), "Output value is not iterable; Node model: {}, Values:{}".format(node.model, values)
        output = { out: value for out, value in zip(node.output, values) }
        self.variables.update(**output)


    def start(self, inputs):
        start_nodes = []
        clear_network_state(self.network)
        # Search start nodes
        for _, node in self.network.nodes(data='node'):
            for name in inputs:
                node.ready(name)
            if node.is_already():
                start_nodes.append(node)

        # Invoke start nodes
        for node in start_nodes:
            output = self.invoke(node)
            self.update_variables(node, output)
        
        # Trace all path of processing flow.
        next_node_names = [ node.name for node in start_nodes ]
        while len(next_node_names) != 0:
            next_nodes = self.walk(next_node_names)
            next_node_names = [ node.name for node in next_nodes ]

    def walk(self, start_node_names):
        already_nodes = []
        for node_name in start_node_names:
            node = self.network.nodes[node_name]['node']
            # if node.is_iterable(): # Iterable nodes found.
            #     next_node = IteratingNode(node, self)
            #     self.network.add_node(next_node.name, node=next_node)
            #     already_nodes.append(next_node)
            # else:
            already_nodes.extend(self.search_next_node(node))

        for node in already_nodes:
            try:
                output = self.invoke(node)
            except Exception:
                print('Uncaught exception occured: {}, {}'.format(node.name, node.model))
                raise
            self.update_variables(node, output)

        return already_nodes

    def search_next_node(self, node):
        already_nodes = []

        if (isinstance(node, IteratingNodeWrapper)):
            edges = self.network.adj[node.name]
            for dst_name, dst_prop in edges.items():
                if 'iteration' in dst_prop:
                    dst_node = self.network.nodes[dst_name]['node']
                    if dst_prop['iteration']:
                        already_nodes.append(dst_node)
                        return already_nodes

                    self.variables.update(dict(zip( node.output, dst_node.model.get_output() ))) # Update variable by iteartion result.
        
        for next_node_name in self.network[node.name]: # Iterate about children nodes.
            edge = self.network.edges[node.name, next_node_name]
            
            if 'input' in edge:   # If true, this egde is the flow of egde.
                next_node = self.network.nodes[next_node_name]['node']
                for i in edge['input']:
                    next_node.ready(i)
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

        in_values = []
        for var in node.input:
            assert var in variables, 'Unknown variable: ' + var
            value = variables[var]
            if logger.isEnabledFor(DEBUG) and hasattr(value, 'shape'):
                logger.debug('Arguments: {}, {}'.format(var, value.shape))
            in_values.append(value)

        out = node(*in_values, **node.args)
        if not isinstance(out, (list, tuple)):
            out = [ out ]

        return out

class NetworkBuilder:
    def __init__(self, graph):
        self.graph = graph
        self.start_nodes = []
        self.already_node_names = []
        clear_network_state(self.graph)

    def ready_node(self, input, node, source):
        if input not in node.input:
            return

        node.ready(input)

        if source is None: # If true, source node is input from dataset.
            return
        
        key = (source, node.name)
        if key in self.graph.edges:
            nx.get_edge_attributes(self.graph, 'input')[key].append(input)
        else:
            self.graph.add_edge(*key, input=[input])

    def build(self, input_list, source_node = None):
        already_nodes = []
        
        for node in [ n for _, n in self.graph.nodes(data='node')]:
            if node.name in self.already_node_names:
                continue

            for input_ in input_list:
                self.ready_node(input_, node, source_node)

            if node.is_already():
                if node.is_iterable(): # If iterable node, setup to loop edge
                    self.graph.add_edge(node.iteration_from_node, node.name, iteration=True)
                    self.graph.add_node(
                        node.iteration_from_node, 
                        node=IteratingNodeWrapper(node, self.graph.nodes[node.iteration_from_node]['node'], self.graph)
                        ) # Replace node for iteration.
                already_nodes.append(node)

        if source_node is None:
            self.start_nodes = already_nodes
            
        self.already_node_names.extend( [ node.name for node in already_nodes ] )
         
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

    def get_node(self, name):
        return self.network.nodes[name]['node']

    def get_network_dict(self):
        return dict(self.network.nodes(data='node'))
        
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
            for _, node in self.network.nodes(data='node'): 
                if name in node.output:
                    found_node = node
                    break
            return found_node

        not_reached_node = search_node(not_reached)

        if not_reached_node is None or isinstance(not_reached_node, tuple):
            return []

        found_nodes = []

        for name, is_ready in not_reached_node.is_already_.items():
            if not is_ready:
                found_node = search_node(name)
                if found_node is not None: # If found_node is None, A searching path reached any input.
                    found_nodes.append(found_node)
                    found_nodes.extend(self.validate_network(name))

        return found_nodes

    def update(self):
        for node in self.updatable_node:
            node.update()

    def __call__(self, mode='train', **inputs):
        assert all((name in inputs for name in self.input_list)) or mode == 'test', \
            'Input requirement is not satisfied. (Inputs: {}, Input requirement: {}])'.format(list(inputs.keys()), self.input_list)
        #assert len(self.network.edges) != 0, "Network is not initialized. you need to invoke build network."
        assert len(self.network.nodes) > 0, "Network node is empty."
        if len(self.network.edges) == 0:
            self.build_network()

        self.variables = {}
        self.variables.update(inputs)
        self.mode = self.mode_list[mode]

        walker = NetworkWalker(self.network, self.mode, self.variables)
        walker.start(inputs)
        
