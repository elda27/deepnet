import deepnet.config
from deepnet import process
from deepnet.utils import network, visualizer
import hashlib
from datetime import datetime
from time import sleep
import glob
import os.path
import copy
from chainer.serializers import load_npz

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

    return _created_process[name]['proc']

def build_networks(config, step=None):
    """Construct network and visualizers from the configuration.
    
    Args:
        config (dict): Configuration of network and visualization after variable expansion.
        step (int, optional): Current step (Default: None)
    
    Raises:
        KeyError: If requirement keys are not available.
    
    Returns:
        [network.NetworkManager, list[visualizer.Visualizer]]: Constructed NetworkManager and list of Visualizer objects.
    """
    config = copy.deepcopy(config) # not to affect changes in the other function.
    network_manager = network.NetworkManager(config['config']['input'])

    # Geenerate process
    for process_config in config['process']:
        build_process(process_config)

    # Generate processing network
    for network_conf in config['network']:
        if step in network_conf['step'] and step is not None:
            selector_step = network_conf['step']
            if ( isinstance(selector_step, list) and step in selector_step ) or
               ( step == selector_step ):
                continue

        if 'label' not in network_conf:
            sleep(1e-6)
            now = str(datetime.now()).encode('ascii')
            network_conf['label'] = hashlib.md5(now).hexdigest()

        proc = None
        updatable = False
        process_name = network_conf.pop('process')
        process_names = process_name.split('.')
        if process_names[0] in _created_process: # If process is exist.
            # :TODO: The updatable parameter use for optimization of the network.
            #        But this implementation need after implementation of optimization configuration
            #        on the config field. (Maybe the implmentation of staging optimization too)

            registed_proc = _created_process[process_names[0]] # registed_proc contains 'proc' and 'proerty', and so on.
            proc = registed_proc['proc']
            proc = deepnet.utils.get_field(proc, process_names[1:])
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

        network_manager.add(
            network_conf.pop('label'), 
            network.NetworkNode(
                network_conf.pop('input'),
                network_conf.pop('output'),
                proc, 
                updatable=updatable,
                training=network_conf.pop('train', True),
                validation=network_conf.pop('valid', True),
                test=network_conf.pop('test', True),
                args=network_conf
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

_registed_initialize_field = {}

def register_initialize_field(name):
    def _register_initialize_field(func):
        _registed_initialize_field[name] =func
        return func
    return _register_initialize_field

def initialize_networks(log_root_dir, step_index, config):
    """Initialize network
    
    Args:
        log_root_dir (str): Root directory of the log.
        stage_index (int): Index of the learning step.
        config (dict): Configuration of the network.
    """

    if 'initialize' not in config:
        return

    initialize_fields = config['initialize']
    for field in initialize_fields:
        _registed_initialize_field[field['mode']](field, log_root_dir, step_index)
    
@register_initialize_field('load')
def initialize_prelearned_model(field, log_root_dir, step_index):
    if 'from_step' in field:
        step_index = field['from_step']

    name = field['name']
    created_model = _created_process[name]
    archive_filename = list(
        glob.glob(os.path.join(log_root_dir, 'model_step' + str(step_index), name + '_*.npz'))
    )[-1]
    
    load_npz(archive_filename, created_model)
    

@register_initialize_field('share')
def shared_layer(field, log_root_dir, step_index):
    get_field = deepnet.utils.get_field
    to_fields = field['to']
    from_fields = field['from']
    freeze_layer = field.get('freeze', False)

    if not isinstance(to_fields, list):
        to_fields = [ to_fields ]

    if not isinstance(from_fields, list):
        from_fields = [ from_fields ]

    for to_field, from_field in zip():
        to_names = to_field.split('.')
        from_names = from_field.split('.')

        from_model = _created_process[from_names[0]]
        to_model = _created_process[to_names[0]]

        from_layer = get_field(from_model, from_names[1:])
        to_parent_layer = get_field(to_model, to_names[1:-1])

        with to_model.init_scope():
            if field.get('deepcopy', False):
                if hasattr(to_parent_layer, to_names[-1]):
                    raise AttributeError('Destination layer doesn\'t have attribute: {}'.format(to_names[-1]))
                from_layer.copyparams(getattr(to_parent_layer, to_names[-1]))
                
                if hasattr(to_parent_layer, 'layers'):
                    to_parent_layer.layers[to_names[-1]] = to_layer
            else:
                setattr(to_parent_layer, to_names[-1], from_layer)
                if hasattr(to_parent_layer, 'layers'):
                    to_parent_layer.layers[to_names[-1]] = from_layer
        