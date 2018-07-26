from contextlib import contextmanager

_global_config = {
    'gpu_id': -1,
    'batch_size': 5,
}

def set_global_config(key, value):
    _global_config[key] = value

def get_global_config(key):
    return _global_config[key]

@contextmanager
def bind_config(key, value):
    tmp = _global_config[key]
    _global_config[key] = value
    yield
    _global_config[key] = tmp
