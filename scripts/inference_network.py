import auto_init
import argparse
from deepnet.utils.network import NetworkNode
from deepnet.utils import mhd
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
import tqdm

def main():
    parser = build_arguments()
    args = parser.parse_args()

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()

    assert args.step_index > 0

    dataset_config = deepnet.config.load(args.dataset_config)
    test_index = deepnet.utils.parse_index_file(args.test_index)

    test_dataset = deepnet.utils.dataset.GeneralDataset(dataset_config, test_index)
    if test_index is None:
        test_index = test_dataset.case_names

    log_dir = deepnet.utils.get_log_dir(args.log_root_dir, args.log_index)

    visualize_dir = os.path.join(log_dir, 'visualize_step{}'.format(args.step_index))
    archive_dir = os.path.join(log_dir, 'model_step{}'.format(args.step_index))
    param_dir = os.path.join(log_dir, 'param_step{}'.format(args.step_index))
    test_dir = os.path.join(log_dir, 'test_step{}'.format(args.step_index))
    log_dirs = {
        'root': log_dir,
        'visualize': visualize_dir,
        'archive': archive_dir,
        'param': param_dir,
        'test': test_dir
    }

    if args.output_dir is None:
        args.output_dir = test_dir

    # load network configuration
    network_config = None
    if args.network_config is None:
        with open(os.path.join(log_dirs['param'], 'network_config.json')) as fp:
            network_config = json.load(fp)
    else:
        network_config = deepnet.config.load(args.network_config, is_variable_expansion=False)
        network_config = update_log_dir(network_config, log_dirs) # Update log directory
        network_config = deepnet.config.expand_variable(network_config)
    network_manager, visualizers = deepnet.core.registration .build_networks(network_config)

    for name, proc in deepnet.core.registration ._created_process.items():
        if proc in deepnet.core.registration ._updatable_process:
            continue
        model_list = list(glob.glob(os.path.join(archive_dir, name + '_*.npz')))
        if len(model_list) == 0:
            raise ValueError('Model not found: {} in {}'.format(name, archive_dir))
        model_filename = model_list[-1]
        chainer.serializers.load_npz(model_filename, proc['proc'])
        proc['proc'].to_gpu()

    # Parse save list
    save_image_list = {}
    for string in args.save:
        pos = string.find(':')
        if pos == -1:
            raise ValueError('Bad format save image list: {}'.format(string))
        save_image_list[string[:pos]] = string[pos + 1:]

    # Start inference.
    variables = {}
    encoded_codes_list = []
    index_list = { idx: 0 for idx in test_index }
    test_iterator = chainer.iterators.MultiprocessIterator(test_dataset, args.batch_size, repeat=False, shuffle=False)

    with chainer.no_backprop_mode():
        for i, batch in tqdm.tqdm(enumerate(test_iterator), total=len(test_dataset) // args.batch_size):
            variables['__iteration__'] = i
            variables['__test_iteration__'] = i
            if args.n_max_test_iter is not None and i >= args.n_max_test_iter:
                break

            input_vars = deepnet.utils.batch_to_vars(batch)
            # Inference
            for j, stage_input in enumerate(input_vars):
                network_manager(mode='test', **stage_input)
                variables['__stage__'] = j
                variables.update(network_manager.variables)
            
            # Save images
            save_images(args.output_dir, variables, save_image_list, index_list)
            

def save_images(output_dir, variables, save_image_list, index_list):
    for image_name, output_filename in save_image_list.items():
        image = deepnet.utils.unwrapped(variables[image_name])
        spacing = variables['spacing']
        #if image.ndim == 4:
        for i in range(image.shape[0]):
            case_name = variables['case_name'][i]
            variables['__index__'] = index_list[case_name]
            index_list[case_name] += 1
            # make output dir
            current_output_dir = os.path.join(output_dir, case_name)
            os.makedirs(current_output_dir, exist_ok=True)
            
            # save images
            current_output_filename = os.path.join(current_output_dir, output_filename.format(**variables))
            save_image(current_output_filename, image[i], spacing[i])
        #else:
        #    current_output_dir = os.path.join(output_dir, variables['case_name'])
        #    os.makedirs(current_output_dir, exist_ok=True)
        #    save_image(output_filename.format(**variables), image[i], spacing)

def save_image(output_filename, image, spacing):
    if spacing is not None and len(spacing) < image.ndim:
        spacing = tuple(spacing) + (1,) * (image.ndim - len(spacing))
    deepnet.utils.visualizer.save_image(output_filename, image, spacing)

def build_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--batch-size', type=int, default=5, help='batch size')

    parser.add_argument('--dataset-config', type=str, required=True, help='A dataset configuraiton written by extended toml format.')
    parser.add_argument('--network-config', type=str, default=None)
    
    parser.add_argument('--test-index', type=str, help='training indices text')
    parser.add_argument('--n-max-test-iter', type=int, default=60000, help='Max iteration of train.')

    parser.add_argument('--log-root-dir', type=str, default='./log/')
    parser.add_argument('--log-index', type=int, default=None, help='Log direcotry index for training.')
    parser.add_argument('--step-index', type=int, default=1, help='step index')

    parser.add_argument('--save', type=str, required=True, nargs='+', help='Paired string of variable name and save format. Save format can be used variable such as __index__.')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--mode', type=str, default='none')

    return parser

def str2bool(string):
    string = string.lower()
    if string in ('on', 'true', 'yes'):
        return True
    elif string in ('off', 'false', 'no'):
        return False
    else:
        raise ValueError('Unknown flag value: {}'.format(string))
    
def update_log_dir(network_config, log_dir):
    network_config['log_dir'] = log_dir['root']
    network_config['visualize_dir'] = log_dir['visualize']
    network_config['archive_dir'] = log_dir['archive']
    network_config['param_dir'] = log_dir['param']
    return network_config

if __name__ == '__main__':
    main()
