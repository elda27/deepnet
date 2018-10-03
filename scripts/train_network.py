import auto_init
import argparse
from corenet import NetworkNode, UpdatableNode
import deepnet
import toml
import numpy as np
try:
    import cupy as cp
except ImportError:
    pass
import chainer
import chainer.functions as F
from chainer import cuda
import datetime
import glob
import math
import os
import os.path
from functools import reduce
from itertools import cycle
import json
import log_util
from logging import getLogger

def main():
    logger = getLogger(__name__)

    parser = build_arguments()
    args = parser.parse_args()

    logger.debug(vars(args))

    use_gpu = False
    if len(args.gpu) > 1 or args.gpu[0] >= 0:
        use_gpu = True
        cuda.get_device(args.gpu[0]).use()

    assert len(args.batch_size) == len(args.gpu)
    assert args.step_index > 0

    deepnet.core.config.set_global_config('gpu_id', args.gpu)
    deepnet.core.config.set_global_config('batch_size', args.batch_size)

    # Load configs
    dataset_config = deepnet.config.load(args.dataset_config)

    train_index = deepnet.utils.parse_index_file(args.train_index, 0.9)
    valid_index = deepnet.utils.parse_index_file(args.valid_index, None)

    # Construct dataset
    if args.enable_dataset_extension:
        train_dataset = deepnet.utils.dataset.CachedDataset(dataset_config, train_index, mode='train')
        valid_dataset = deepnet.utils.dataset.CachedDataset(dataset_config, valid_index, mode='valid')
    else:
        train_dataset = deepnet.utils.dataset.GeneralDataset(dataset_config, train_index)
        valid_dataset = deepnet.utils.dataset.GeneralDataset(dataset_config, valid_index)

    # Setup directories
    log_dir = log_util.get_training_log_dir(
        args.log_root_dir, args.log_index, args.step_index, 
        opt_name=dataset_config['config'].get('exp_name') if args.exp_name is None else args.exp_name
        )

    visualize_dir = os.path.join(log_dir, 'visualize_step{}'.format(args.step_index))
    archive_dir = os.path.join(log_dir, 'model_step{}'.format(args.step_index))
    param_dir = os.path.join(log_dir, 'param_step{}'.format(args.step_index))
    log_dirs = {
        'root': log_dir,
        'visualize': visualize_dir,
        'archive': archive_dir,
        'param': param_dir
    }

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(visualize_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    os.makedirs(param_dir, exist_ok=True)

    # Construct network
    network_config = load_network_config(args.network_config, args.hyper_param, log_dirs)
    network_manager, visualizers = deepnet.core.build_networks(network_config)

    # Initialize network

    for init_field in network_config.get('initialize', []):
        deepnet.core.initialize_networks(**init_field)

    # Setup post processor
    postprocessor = deepnet.utils.postprocess.PostProcessManager(network_config.get('postprocess', []))

    # Setup logger
    logger = [ 
        deepnet.utils.logger.CsvLogger(
            os.path.join(log_dir, 'log_step{}.csv'.format(args.step_index)), 
            network_config['config']['logging'],
            network_config['config'].get('logging_weights', [])
            ) 
        ]
    archive_nodes = network_config['config']['archive_nodes']
    optimizing_loss = network_config['config']['optimizing_loss']
    write_architecture_loss = (
        os.path.join(log_dirs['archive'], 'model.png'), 
        network_config['config'].get('archive_loss')
    )
    # Setup optimizer
    optimizers = []
    optimizer = chainer.optimizers.Adam(args.lr_rate)
    for model in deepnet.core.get_updatable_process_list():
        if not issubclass(type(model), chainer.Chain):
            continue
        if use_gpu:
            model.to_gpu()
        optimizer.setup(model)
    optimizers.append(optimizer)

    # Freeze to update layer
    for layer_name in network_config['config'].get('freezing_layer', []):
        layers = layer_name.split('.')
        model = deepnet.core.registration.get_process(layers[0])
        deepnet.utils.get_field(model, layers[1:]).disable_update()

    # Save variables
    with open(os.path.join(param_dir, 'args.json'), 'w+') as fp:
        json.dump(vars(args), fp, indent=2)

    with open(os.path.join(param_dir, 'network_config.json'), 'w+') as fp:
        json.dump(network_config, fp, indent=2)
    
    ## Dump network architecture
    with open(os.path.join(param_dir, 'network_architectuire.json'), 'w+') as fp:
        json_dict = dict(
            network = { 
                    name: dict(
                        input= node.input,
                        output= node.output,
                        updatable= node.update_variable if issubclass(type(node), UpdatableNode) else None,
                        model= str(node.model),
                        attr = { key: str(value) for key, value in node.attrs.items()},
                        args= { name: str(node.args) for name, arg in node.args.items() }
                    )
                     for name, node in network_manager.network.items()
                }
        )
        json.dump(json_dict, fp, indent=2)

    # Start training. 
    train_config = vars(args)
    train_config['progress_vars'] = [ '{}:.3f'.format(loss) for loss in optimizing_loss ]

    optimizer_dict = { loss: optimizer for loss, optimizer in zip(optimizing_loss, optimizers) }

    if args.debug:
        iterator_type = chainer.iterators.SerialIterator
    else:
        iterator_type = chainer.iterators.MultiprocessIterator

    trainer = deepnet.utils.trainer.Trainer(
        network=network_manager,
        train_iter=iterator_type(train_dataset, sum(args.batch_size), shuffle=True, repeat=True),
        valid_iter=iterator_type(valid_dataset, sum(args.batch_size), shuffle=False, repeat=False),
        visualizers=visualizers,
        optimizer=optimizer_dict,
        logger=logger,
        archive_dir=archive_dir,
        archive_nodes=archive_nodes,
        train_config=train_config,
        postprocessor=postprocessor,
        redirect=parse_redirect_string(args.redirect),
        architecture_loss=write_architecture_loss
    )

    trainer.train()
    print(log_dir)


def build_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, nargs="*", default=[0], help='gpu id')
    parser.add_argument('--batch-size', type=int, nargs="*", default=[5], help='batch size')

    parser.add_argument('--exp-name', type=str, default=None, help='Experiments name.'
        'If None, this value will be exp_name in the dataset config.'
        )

    parser.add_argument('--dataset-config', type=str, required=True, help='A dataset configuraiton written by extended toml format.')
    parser.add_argument('--enable-dataset-extension', action='store_true', help='Activate extensions for dataset (It enables caching dataset too.)')
    parser.add_argument('--network-config', type=str, required=True, help='A network configuraiton written by extended toml format.')
    parser.add_argument('--hyper-param', type=str, default=None, nargs='*', help='Set hyper parameters defined on network config. (<param name>:<value>)')
    
    #parser.add_argument('--n-channel', type=int, default=14, help='n channel of input data')
    #parser.add_argument('--n-layers', type=int, default=5, help='n channel of input data')

    parser.add_argument('--train-index', type=str, default=None, help='training indices text')
    parser.add_argument('--valid-index', type=str, default=None, help='validation indices text')
    parser.add_argument('--lr-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n-max-train-iter', type=int, default=60000, help='Max iteration of train.')
    parser.add_argument('--n-max-valid-iter', type=int, default=None, help='Max iteration of validation.')
    parser.add_argument('--n-valid-step', type=int, default=5000, help='Step of validation every this iteration.')

    parser.add_argument('--log-root-dir', type=str, default='./log/')
    parser.add_argument('--log-index', type=int, default=None, help='Log direcotry index for training.')
    parser.add_argument('--step-index', type=int, default=1, help='step index')

    parser.add_argument('--redirect', type=str, default=[], nargs='*', help='To redirect input variables.')
    parser.add_argument('--debug', action='store_true', default=False, help='If true, this session will start single process.')

    return parser


def str2bool(string):
    string = string.lower()
    if string in ('on', 'true', 'yes'):
        return True
    elif string in ('off', 'false', 'no'):
        return False
    else:
        raise ValueError('Unknown flag value: {}'.format(string))


def parse_hyper_parameter(params, defined_params):
    if params is None:
        return defined_params

    result_params = {}
    for param in params:
        pos = param.find(':')
        target = param[:pos]
        value = param[pos+1:]
        
        assert target in defined_params, 'Unknown hyper parameter: {}'.format(target)

        type_ = type(defined_params[target])
        try:
            result_params[target] = type_(value)
        except:
            raise TypeError('Invalid value detected on the cast:{}, str->{}'.format(value, type_))
    return result_params


def load_network_config(config_filename, hyper_param, log_dirs):
    kwargs = dict(hyper_param=hyper_param, log_dirs=log_dirs)
    network_config = deepnet.config.load(config_filename, is_variable_expansion=False)
    network_config = expand_include(network_config, **kwargs)
    network_config = update_log_dir(network_config, log_dirs) # Update log directory
    network_config['hyper_parameter'].update(parse_hyper_parameter(hyper_param, network_config['hyper_parameter']))
    network_config = deepnet.config.expand_variable(network_config)
    return network_config


def expand_include(config, **kwargs):
    if 'include' not in config:
        return config

    include_config = load_network_config(config['include'], **kwargs)

    return config.get('network_before', []) + include_config['network'] + config.get('network_after', [])


def parse_redirect_string(strings):
    result = {}
    for string in strings:
        src, dst = string.split(':')
        result[src] = dst
    return result

def update_log_dir(network_config, log_dir):
    network_config['log_dir'] = log_dir['root']
    network_config['visualize_dir'] = log_dir['visualize']
    network_config['archive_dir'] = log_dir['archive']
    network_config['param_dir'] = log_dir['param']
    return network_config

if __name__ == '__main__':
    main()
