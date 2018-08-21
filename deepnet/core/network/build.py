import deepnet.config
from deepnet import process
from deepnet.utils import visualizer
from deepnet.core.registration import register_argument_wrapper, generate_network, exist_process, get_registered_process
from deepnet.core.config import get_global_config
import corenet
import hashlib
from datetime import datetime
from time import sleep
import glob
import os.path
import copy
from chainer.serializers import load_npz

import warnings

_created_process = {}
_updatable_process = []

def get_updatable_process_list():
    return _updatable_process

def build_process(process_config):
    """Construct process from process tag of the configuration.
    
    Args:
        process_config (dict): An element of process tag of the configuration.
    """

    
    assert 'label' in process_config, 'Key error: ' + str(process_config)
    assert 'type' in process_config, 'Key error: ' + str(process_config)
    
    type_name = process_config.pop('type')
    label     = process_config.pop('label')
    update    = process_config.pop('update', None) 

    assert label not in _created_process, 'Process label is duplicated:' + label

    proc = generate_network(type_name, **process_config)

    _created_process[label] = {
        'proc': proc,
        'property': process_config,
        'update': update,
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
        [corenet.NetworkManager, list[visualizer.Visualizer]]: Constructed NetworkManager and list of Visualizer objects.
    """
    config = copy.deepcopy(config) # not to affect changes in the other function.
    network_manager = corenet.NetworkManager()

    # Geenerate process
    for process_config in config['process']:
        build_process(process_config)

    # Generate processing network
    for network_conf in config['network']:
        if step in network_conf.get('step', []) and step is not None:
            selector_step = network_conf['step']
            if ( isinstance(selector_step, list) and step in selector_step ) or \
               ( step == selector_step ):
                continue

        if 'label' not in network_conf:
            network_conf['label'] = corenet.get_unique_label()

        node_type = corenet.NetworkNode
        proc = None
        updatable = False
        process_name = network_conf.pop('process')
        process_names = process_name.split('.')
        if process_names[0] in _created_process: # If process is exist.
            # :TODO: The updatable parameter use for optimization of the network.
            #        But this implementation need after implementation of optimization configuration
            #        on the config field. (Maybe the implmentation of staging optimization too)

            registered_proc = _created_process[process_names[0]] # registered_proc contains 'proc' and 'proerty', and so on.
            proc = registered_proc['proc']
            proc = deepnet.utils.get_field(proc, process_names[1:])
            updatable = registered_proc['update']
            _updatable_process.append(proc)
            if not updatable:
                warnings.warn('A defined network is used on network stream but an not updatable network. {}'.format(process_names[0]))
            
            if len(get_global_config('gpu_id')) > 1:
                node_type = network.ParallelNetworkNode

        elif exist_process(process_name):
            proc = get_registered_process(process_name)
            updatable = False

        else:
            raise KeyError('Unknown process:{process} <input: {input}, output: {output}>'.format(
                process=process_name, 
                input=network_conf['input'],
                output=network_conf['output'],
                ))

        network_manager.add_node( 
            node_type(
                network_conf.pop('label'),
                proc, 
                network_conf.pop('input'),
                network_conf.pop('output'),
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

@register_argument_wrapper('network')
def wrap_network_name(network_name):
    names = network_name.split('.')
    model = get_process(names[0])
    return deepnet.utils.get_field(model, names[1:])
