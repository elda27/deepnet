import deepnet.config
from deepnet import process
from deepnet.utils import network, visualizer
import hashlib
from datetime import datetime
from time import sleep
import copy

_registed_network = {}
_created_process = {}
_updatable_process = []

def register_network(name):
    """Decorator function to register network by label.
    
    Args:
        name (str): label of registering network
    """

    assert name not in _registed_network, 'Registering key name is exist. ' + name
    def _register_network(klass):
        _registed_network[name] = klass
        return klass
    return _register_network

def generate_network(name, **kwargs):
    """Generate registed network from name
    
    Args:
        name (str): Label of registered by register_network function.
        kwargs : Network arguments.
    
    Returns:
        object: Registed network coresponding name
    """

    return _registed_network[name](**kwargs)

def build_process(process_config):
    """Construct process from process tag of the configuration.
    
    Args:
        process_config (dict): An element of process tag of the configuration.
    """

    
    assert 'label' in process_config, 'Key error: ' + str(process_config)
    assert 'type' in process_config, 'Key error: ' + str(process_config)
    
    type_name = process_config['type']
    label = process_config['label']

    assert label not in _created_process, 'Process label is duplicated:' + label

    proc = generate_network(type_name, **process_config)

    _created_process[label] = {
        'proc': proc,
        'property': process_config
    }

def get_process(name):
    """Get created process.
    
    Args:
        name (str): The name of process.
    
    Returns:
        any: Create process.
    """

    return _created_process[name]

def build_networks(config):
    """Construct network and visualizers from the configuration.
    
    Args:
        config (dict): Configuration of network and visualization after variable expansion.
    
    Raises:
        KeyError: If requirement keys are not available.
    
    Returns:
        [network.NetworkManager, list[visualizer.Visualizer]]: Constructed NetworkManager and list of Visualizer objects.
    """

    network_manager = network.NetworkManager(config['config']['input'])

    # Geenerate process
    for process_config in config['process']:
        build_process(process_config)

    # Generate processing network
    for network_conf in config['network']:
        if 'label' not in network_conf:
            sleep(1e-6)
            now = str(datetime.now()).encode('ascii')
            network_conf['label'] = hashlib.md5(now).hexdigest()

        proc = None
        updatable = False
        process_name = network_conf['process']
        if process_name in _created_process: # If process is exist.
            registed_proc = _created_process[process_name]
            proc = registed_proc['proc']
            updatable = 'update' in registed_proc['property']
            _updatable_process.append(proc)

        elif process_name in process._registed_process:
            proc = process._registed_process[process_name]
            updatable = False

        else:
            raise KeyError('Unknown process:{process} <input: {input}, output: {output}>'.format(
                process=process_name, 
                input=network_conf['input'],
                output=network_conf['output'],
                ))

        args = copy.deepcopy(network_conf)
        try:
            del args['label'], args['input'], args['output'], args['process']
        except:
            pass

        network_manager.add(
            network_conf.pop('label'), 
            network.NetworkNode(
                network_conf.pop('input'),
                network_conf.pop('output'),
                proc, 
                updatable=updatable,
                training=network_conf.pop('train', False),
                validation=network_conf.pop('valid', False),
                test=network_conf.pop('test', False),
                args=args
                )
            )
    
    # Geneerate visualizer
    visualizers = []
    for network_conf in config['visualize']:
        assert 'type' in network_conf, \
            'Key error: (Key: type, Dict:{})'.format(str(network_conf))
        
        type_name = network_conf['type']
        del network_conf['type']
        visualizers.append(visualizer.create_visualizer(type_name)(**network_conf))

    return network_manager, visualizers
