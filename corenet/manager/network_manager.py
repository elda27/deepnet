from collections import OrderedDict
from .process_pointer import ProcessPointer
from corenet.node.updatable_node import UpdatableNode

class NetworkManager:
    def __init__(self):
        self.process_list = OrderedDict()
        self.pointer = ProcessPointer(self.process_list)
        self.variables = {}

    def get_node(self, name):
        assert name in self.process_list
        return self.process_list[name]

    def add_node(self, node):
        assert node.name not in self.network,\
            'Duplicating label name: {}({})<{}>'.format(node.name, node, str({ key: str(node) for key, node in self.network.items() }))
        self.network[node.name] = node

    def build_network(self):
        self.pointer.sync()

    def validate_network(self):
        unreached = []
        for node in self.pointer.backward():
            unreached.append(str(node))
        return list(reversed(unreached))

    def update(self):
        updatables = [ 
            node.update() for node in self.pointer.forward(0) 
            if issubclass(node, UpdatableNode)
        ]
        for i in range(3):
            for updatable in updatables:
                next(updatable)

    def invoke(self, node):
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

    def is_runtime(mode, node):
        value = node.attrs.get(mode, True)
        return value if isinstance(value, bool) else False

    def update_variables(self, node, values):
        assert isinstance(values, (tuple, list)), \
            'Output value is not iterable; Node model: {}, Values:{}'.format(node.model, values)
        output = { out: value for out, value in zip(node.output, values) }
        self.variables.update(**output)

    def __call__(self, mode='train', **inputs):
        assert len(self.network.nodes) > 0, "Network node is empty."
        if len(self.network.edges) == 0:
            self.build_network()

        self.variables = {}
        self.variables.update(inputs)

        assert all((name in inputs for name in self.input_list)) or mode == 'test', \
            'Input requirement is not satisfied. (Inputs: {}, Input requirement: {}])'.format(list(inputs.keys()), self.input_list)

        for node in self.pointer.forward(0):
            if not self.is_runtime(mode, node):
                continue
            try:
                output = self.invoke(node)
            except Exception:
                print('Uncaught exception occured: {}, {}'.format(node.name, node.model))
                raise
            self.update_variables(node, output)

