from enum import Enum
from .network_node import NetworkNode
from .updatable_node import UpdatableNode

class NodeType(Enum):
    Default   = 'default'
    Updatable = 'updatable'

_declared_processors = {}

def declare_node_type(node_type, **kwargs):
    def _declare_node_type(klass):
        if node_type == NodeType.Updatable:
            _declare_node_type[klass] = (UpdatableNode, kwargs)

def make_node(model, **kwargs):
    klass = type(model)
    node_type = NetworkNode
    if klass in _declared_processors:
        node_type, args = _declared_processors[klass]
        kwargs.update(args)
    
    return node_type(model = model, **kwargs)
