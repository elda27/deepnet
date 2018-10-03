from collections import OrderedDict
from .process_pointer import ProcessPointer
from corenet.node.updatable_node import UpdatableNode

from logging import getLogger, DEBUG

logger = getLogger(__name__)


class NetworkManager:
    def __init__(self):
        self.process_list = OrderedDict()
        self.pointer = ProcessPointer(self.process_list)
        self.variables = {}

    @property
    def network(self):
        return self.process_list

    def get_node(self, name):
        assert name in self.process_list
        return self.process_list[name]

    def add_node(self, node):
        assert node.name not in self.process_list,\
            'Duplicating label name: {}({})<{}>'.format(node.name, node, str(
                {key: str(node) for key, node in self.network.items()}))
        self.process_list[node.name] = node

    def build_network(self):
        self.pointer.sync()

    def validate_network(self):
        unreached = []
        for node in self.pointer.backward():
            unreached.append(str(node))
        return list(reversed(unreached))

    def update(self):
        updatables = []
        for node in self.pointer.forward(0):
            if not issubclass(type(node), UpdatableNode):
                continue

            if node.update_variable is None:
                node.update_before()
                continue

            if node.update_variable not in self.variables:
                unreached = self.validate_network(node.update_variable)
                raise ValueError(
                    'Unreached loss computation.\nFollowing list is not reached nodes: \n' +
                    '\n'.join([str(n) for n in unreached])
                )
            updatables.append(node.update(self.variables))

        for _ in range(3):
            for updatable in updatables:
                next(updatable)

    def invoke(self, node):
        in_values = []
        for var in node.input:
            assert var in self.variables, 'Unknown variable: ' + var
            value = self.variables[var]
            if logger.isEnabledFor(DEBUG):
                if hasattr(value, 'shape'):
                    logger.debug('Arguments: {}, {}'.format(var, value.shape))
                else:
                    logger.debug('Arguments: {}, {}'.format(var, value))
            in_values.append(value)

        out = node(*in_values)
        if not isinstance(out, (list, tuple)):
            out = [out]

        return out

    def is_runtime(self, mode, node):
        value = node.attrs.get(mode, True)
        return value if isinstance(value, bool) else False

    def update_variables(self, node, values):
        """Update storing variables.

        Args:
            node (NetworkNode): A processed node.
            values (dict): Output variable name and value.
        """

        assert isinstance(values, (tuple, list)), \
            'Output value is not iterable; Node model: {}, Values:{}'.format(
                node.model, values)
        output = {out: value for out, value in zip(node.output, values)}
        self.variables.update(**output)

    def __call__(self, mode='train', **inputs):
        assert len(self.network) > 0, "Network node is empty."
        self.build_network()

        self.variables = {}
        self.variables.update(inputs)

        # assert all((name in inputs for name in self.input_list)) or mode == 'test', \
        #    'Input requirement is not satisfied. (Inputs: {}, Input requirement: {}])'.format(list(inputs.keys()), self.input_list)

        for node in self.pointer.forward(0):
            if not self.is_runtime(mode, node):
                continue
            try:
                output = self.invoke(node)
            except Exception:
                print('Uncaught exception occured: {}, {}'.format(
                    node.name, node.model))
                raise
            self.update_variables(node, output)
