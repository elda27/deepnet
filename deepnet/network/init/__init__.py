import deepnet.config
from deepnet.utils import network, process, visualizer
import hashlib
from datetime import datetime

_registed_network = {}
_created_process = {}

def register_network(name):
    """Decorator function to register network by label.
    
    Args:
        name (str): label of registering network
    """

    assert name in _registed_network, 'Registering key name is exist. ' + name
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

    _created_process[label] = proc

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
            now = bytes(str(datetime.now()))
            network_conf['label'] = hashlib.md5(now).hexdigest()

        proc = None
        updatable = False
        process_name = network_conf['process']
        if process_name in _created_process:
            registed_proc = _created_process[process_name]
            proc = registed_proc['proc']
            updatable = 'update' in registed_proc['property']
        elif process_name in process._registed_process:
            proc = process._registed_process[process_name]
            updatable = False
        else:
            raise KeyError('Unknown process:{process} [input: {input}, output: {output}]'.format(
                process=process_name, 
                input=network_conf['input'],
                output=network_conf['output'],
                ))

        network_manager.add(
            network_conf['label'], 
            network.NetworkNode(
                network_conf['input'],
                network_conf['output'], 
                proc, 
                updatable=updatable
                )
            )
    
    # Geneerate visualizer
    visualizers = []
    for network_conf in config['visualizer']:
        assert 'type' in network_conf, \
            'Key error: (Key: type, Dict:{})'.format(str(network_conf))
        
        type_name = network_conf['type']
        del network_conf['type']
        visualizers.append(visualizer.create_visualizer(type_name)(**network_conf))

    return network_manager, visualizers
