from logging import getLogger

logger = getLogger(__name__)
_registered_process = {}

def add_process(name, func):
    logger.debug('Add process: [{}, {}]'.format(name, func))
    assert name not in _registered_process, \
        "The name of process is existing: {}".format(name)
    _registered_process[name] = func

def exist_process(name):
    return name in _registered_process

def get_registered_process(name):
    return _registered_process[name]

def register_process(name = None):
    def _register_process(func):
        if name is None:
            add_process(func.__name__, func)
        else:
            add_process(name, func)
        return func
    return _register_process

def invoke_process(name, *args, **kwargs):
    return _registered_process[name](*args, **kwargs)


_registered_network = {}
_registered_arguments_wrappers = {}

def add_network(name, klass):
    assert name not in _registered_network, \
        "The name of network processor is existing: {}".format(name)
    _registered_network[name] = klass

def register_network(name, wrap_args = {}):
    """Decorator function to register network by label.
    
    Args:
        name (str): label of registering network
    """

    assert name not in _registered_network, 'Registering key name is exist. ' + name
    def _register_network(klass):
        _registered_network[name] = { "class": klass, "args": wrap_args } 
        return klass
    return _register_network

def register_argument_wrapper(spec_name):
    assert spec_name not in _registered_arguments_wrappers, 'Registering key name is exist. ' + spec_name
    def _register_arguments_wrappers(func):
        _registered_arguments_wrappers[spec_name] = func
        return func
    return _register_arguments_wrappers

def generate_network(name, **kwargs):
    """Generate registered network from name
    
    Args:
        name (str): Label of registered by register_network function.
        kwargs : Network arguments.
    
    Returns:
        object: registered network coresponding name
    """

    try:
        target_network = _registered_network[name]
        # Wrap arguments
        for key, spec_name in target_network['args'].items():
            if key not in kwargs:
                continue
            kwargs[key] = _registered_arguments_wrappers[spec_name](kwargs[key])

        return target_network['class'](**kwargs)
    except TypeError:
        print('Unexpected keyword error:\nThe problem argument is {}'.format(kwargs))
        raise

_registered_initialize_field = {}

def register_initialize_field(name):
    def _register_initialize_field(func):
        _registered_initialize_field[name] =func
        return func
    return _register_initialize_field