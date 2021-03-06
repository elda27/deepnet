from enum import Enum
from functools import wraps

from .network_node import NetworkNode
from .updatable_node import UpdatableNode
from .chainer_node   import ChainerNode
from .iterable_node import IterableNode


_map_processor_type = {
    'default'   : NetworkNode,
    'updatable' : UpdatableNode,
    'chainer'   : ChainerNode,
    'iterable'  : IterableNode,
}
_declared_processors = {}

def declare_node_type(node_type, **kwargs):
    """Declare node type for make_node function.
    
    Args:
        node_type (str): Node type to generate node wrapping any processor.
    """

    def _declare_node_type(klass):
        assert node_type in _map_processor_type, 'Unknown node type:' + node_type
        _declared_processors[klass] = (_map_processor_type[node_type], kwargs)
        return klass
    return _declare_node_type

def make_node(model, **kwargs):
    """Generator function of network node.
    
    Args:
        model (function): Process function
    
    Returns:
        node_type: Generated node
    """

    klass = type(model)
    node_type = NetworkNode
    if klass in _declared_processors:
        node_type, args = _declared_processors[klass]
        kwargs.update(args)
    
    return node_type(model = model, **kwargs)
