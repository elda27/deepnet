import auto_init
import argparse
from deepnet.utils.network import NetworkNode
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

def main():
    parser = build_arguments()
    args = parser.parse_args()

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()

    assert args.stage_index > 0

    dataset_config = deepnet.config.load(args.dataset_config)
    train_index = deepnet.utils.parse_index_file(args.train_index, 0.9)
    valid_index = deepnet.utils.parse_index_file(args.valid_index, None)

    train_dataset = deepnet.utils.dataset.GeneralDataset(dataset_config, train_index)
    valid_dataset = deepnet.utils.dataset.GeneralDataset(dataset_config, valid_index)

    log_dir = get_log_dir(args.log_root_dir, args.log_index, args.stage_index)
    visualize_dir = os.path.join(log_dir, 'visualize_stage{}'.format(args.stage_index))
    archive_dir = os.path.join(log_dir, 'model_stage{}'.format(args.stage_index))
    param_dir = os.path.join(log_dir, 'param_stage{}'.format(args.stage_index))
    log_dirs = {
        'root': log_dir,
        'visualize': visualize_dir,
        'archive': archive_dir,
        'param': param_dir
    }
    os.makedirs(visualize_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    os.makedirs(param_dir, exist_ok=True)

    # load network configuration
    network_config = toml.load(args.network_config)
    network_config = update_log_dir(network_config, log_dirs) # Update log directory
    network_config['hyper_parameter'] = parse_hyper_parameter(args.hyper_param, network_config['hyper_parameter'])
    network_config = deepnet.config.expand_variable(network_config)
    network_manager, visualizers = deepnet.network.init.build_networks(network_config)

    # Setup logger
    logger = [ 
        deepnet.utils.logger.CsvLogger(
            os.path.join(log_dir, 'log_stage{}.csv'.format(args.stage_index)), 
            network_config['config']['logging']
            ) 
        ]
    archive_nodes = network_config['config']['archive_nodes']
    optimizing_loss = network_config['config']['optimizing_loss']

    # Setup optimizer
    optimizers = []
    optimizer = chainer.optimizers.Adam(args.lr_rate)
    for model in deepnet.network.init._updatable_process:
        optimizer.setup(model)
        
    optimizers.append(optimizer)

    # Save variables
    with open(os.path.join(param_dir, 'args.json'), 'w+') as fp:
        json.dump(vars(args), fp, indent=2)

    with open(os.path.join(param_dir, 'network_config.json'), 'w+') as fp:
        json.dump(network_config, fp, indent=2)
    
    ## Dump network architecture
    with open(os.path.join(param_dir, 'network_architectuire.json'), 'w+') as fp:
        json_dict = dict(
            input_list = network_manager.input_list,
            network = { 
                    name: dict(
                        input= node.input,
                        output= node.output,
                        updatable= node.updatable,
                        training= node.training,
                        validation= node.validation,
                        model= str(node.model),
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

    trainer = deepnet.utils.trainer.Trainer(
        network=network_manager,
        train_iter=chainer.iterators.MultiprocessIterator(train_dataset, args.batch_size, shuffle=True, repeat=True),
        valid_iter=chainer.iterators.MultiprocessIterator(valid_dataset, args.batch_size, shuffle=False, repeat=False),
        visualizers=visualizers,
        optimizer=optimizer_dict,
        logger=logger,
        archive_dir=archive_dir,
        archive_nodes=archive_nodes,
        train_config=train_config,
    )

    trainer.train()
    print(log_dir)

def build_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--batch-size', type=int, default=5, help='batch size')

    parser.add_argument('--dataset-config', type=str, required=True, help='A dataset configuraiton written by extended toml format.')
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
    parser.add_argument('--stage-index', type=int, default=1, help='Stage index')

    return parser

def str2bool(string):
    string = string.lower()
    if string in ('on', 'true', 'yes'):
        return True
    elif string in ('off', 'false', 'no'):
        return False
    else:
        raise ValueError('Unknown flag value: {}'.format(string))

def get_log_dir(root_dir, log_index, stage_index, opt_name = ''):
    if log_index is None:
        if stage_index == 1: # 1st stage training and log index is automatically generation
            return get_new_log_dir(root_dir, opt_name=opt_name)
        else:                # After 1st stage training and log directory is user selected.
            return root_dir
    else:
        if stage_index == 1: # 1st stage training and log index user defined.
            return get_new_log_dir(root_dir, start_index=log_index, opt_name=opt_name)
        else:                # After 1st stage training and log index user defined.
            log_dirs = [ log_dir for log_dir in glob.glob(os.path.join(root_dir, str(log_index) + '-*')) if os.path.isdir(log_dir)]
            if len(log_dirs) == 0:
                raise ValueError('Selected index directory is not found: {}\nVerify the root directory: {}'.format(log_index, root_dir))
            return log_dirs[0]
            

def get_new_log_dir(root_dir, opt_name = '', start_index = 0):
    log_dirs = [ log_dir for log_dir in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(log_dir)]
    max_id = -1
    for log_dir in log_dirs:
        log_dir = os.path.basename(log_dir)
        pos = log_dir.find('-')
        if pos == -1:
            continue
        try:
            tmp_max_id = max(max_id, int(log_dir[:pos]))
            if start_index == tmp_max_id:   # Selected index is duplicated so increase index and continue to check duplicating.
                start_index += 1
            elif start_index < tmp_max_id: # Selected index is less than found index so user selected index is not duplicated.
                max_id = start_index
                break
            max_id = tmp_max_id
        except ValueError:
            pass
    
    if max_id <= start_index: # Found index less than use selected index
        max_id = start_index - 1

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    cur_dir = '{}-TIME-{}'.format(max_id + 1, timestamp)
    if opt_name:
        cur_dir = opt_name + '-' + cur_dir

    out = os.path.join(root_dir, cur_dir)
    os.makedirs(out, exist_ok=True)
    return out
    
def parse_hyper_parameter(params, defined_params):
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

def update_log_dir(network_config, log_dir):
    network_config['log_dir'] = log_dir['root']
    network_config['visualize_dir'] = log_dir['visualize']
    network_config['archive_dir'] = log_dir['archive']
    network_config['param_dir'] = log_dir['param']
    return network_config

if __name__ == '__main__':
    main()
