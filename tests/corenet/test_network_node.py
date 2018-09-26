import pytest
import corenet
from itertools import zip_longest

def model_func(*args, **kwargs):
    def _model_func(*_args, **_kwargs):
        assert all([ a1 == a2 for a1, a2 in zip_longest(args, _args)])
        assert all([ k1 == k2 and kwargs[k1] == kwargs[k2] for k1, k2 in zip_longest(kwargs, _kwargs)])
        return _args, _kwargs
    return _model_func

@pytest.mark.parametrize("args, kwargs",
    [[10, 20, 30], dict(a=20, b=30)]
)
def test_network_node(args, kwargs):
    node = corenet.NetworkNode('name', model_func(*args, **kwargs), [], [])
    node(*args, **kwargs)
