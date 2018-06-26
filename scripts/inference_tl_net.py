import argparse
from deepnet import *
from deepnet import network
from deepnet.utils.network import NetworkNode
import numpy as np
try:
    import cupy as cp
except ImportError:
    pass
import chainer
import chainer.functions as F
import datetime
import glob
import math
import os
import os.path
from functools import reduce
from itertools import cycle
import json
import tqdm
from deepnet.utils import mhd

import connections

def main():
    parser = build_arguments()
    args = parser.parse_args()
    log_dir = utils.get_log_dir(args.log_root_dir, args.log_index)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    test_index = parse_index_file(args.test_index)
    print('Test index:', test_index)
    test_dataset = utils.dataset.XpDataset(
        args.dataset_dir, test_index, args.image_type, 
        image=False if args.stage_index == 1 else True, 
        label=False if args.stage_index >= 2 else True)

    param_dir = os.path.join(log_dir, 'param_stage{}'.format(args.stage_index))
    assert os.path.exists(param_dir), 'This directory is not including learned parameters: ' + log_dir
    train_args = load_train_args(param_dir)

    log_dirs = {
        'visualize': os.path.join(log_dir, 'visualize_stage' + str(args.stage_index)),
        'param': param_dir,
    }

    if args.output_root_dir is None:
        args.output_root_dir = os.path.join(log_dir, 'test_stage' + str(args.stage_index))

    # models
    cae_model = network.conv_auto_encoder.ConvolutionalAutoEncoder(
        2, train_args.n_channel, encode_dim=train_args.encode_dims, 
        n_layers=train_args.n_layers,
        dropout=train_args.dropout_mode, use_batch_norm=train_args.use_batch_norm,
        use_skipping_connection = 'none',
        )
    segnet_model = network.tl_net.Segnet(
        2, 1, cae_model.decoder,
        use_skipping_connection=train_args.use_skipping_connection
    )
    
    if args.stage_index == 1:
        model_archive = list(glob.glob(os.path.join(log_dir, 'model_stage' + str(args.stage_index), 'CAE_*.npz')))[-1]
        chainer.serializers.load_npz(model_archive, cae_model)
    if args.stage_index >= 2:
        model_archive = list(glob.glob(os.path.join(log_dir, 'model_stage' + str(args.stage_index), 'Segnet_*.npz')))[-1]
        chainer.serializers.load_npz(model_archive, segnet_model)
    
    # Costruct network model
    train_args.gpu = args.gpu
    network_manager, _ = connections.build_network_for_ribcage(cae_model, segnet_model, train_args, log_dirs, args.stage_index) # TODO: segnetのテストの実装
    #network_manager, visualizers = build_network_for_real_image(cae_model, args, log_dir)
    
    save_image_list = None
    if args.stage_index == 1:
        save_image_list = {
            'gpu_reconstruct_label': '{__index__:08d}_label.mhd'
        }
    elif args.stage_index == 2:
        save_image_list = {
            'gpu_segment_label': '{__index__:08d}_label.mhd'
        }
    elif args.stage_index == 3:
        save_image_list = {
            'gpu_segment_label': '{__index__:08d}_label.mhd'
        }

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

            input_vars = utils.batch_to_vars(batch)
            # Inference
            for j, stage_input in enumerate(input_vars):
                network_manager(mode='test', **stage_input)
                variables['__stage__'] = j
                variables.update(network_manager.variables)
                if args.store_codes:
                    encoded_codes = network_manager.network['CAE'].model['encoder'].stores['fc']
                    encoded_codes_list.append(F.copy(encoded_codes, dst=-1))
            
            # Save images
            save_images(args.output_root_dir, variables, save_image_list, index_list)
            
    if args.store_codes:
        encoded_codes = F.concat(encoded_codes_list, axis=0)
        np.save(os.path.join(args.output_root_dir, 'encoded_codes.np'), encoded_codes)

def save_images(output_dir, variables, save_image_list, index_list):
    for image_name, output_filename in save_image_list.items():
        image = utils.unwrapped(variables[image_name])
        spacing = variables['spacing']
        if image.ndim == 4:
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
        else:
            current_output_dir = os.path.join(output_dir, variables['case_name'])
            os.makedirs(current_output_dir, exist_ok=True)
            save_image(output_filename.format(**variables), image[i], spacing)

def save_image(output_filename, image, spacing):
    #spacing = spacing[::-1]
    #image = np.transpose(image, (2, 1, 0))
    mhd.write(output_filename, image, { 'ElementSpacing': spacing })

def build_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for inference.')
    parser.add_argument('--test-index', type=str, required=True, help='Test indices text')
    parser.add_argument('--dataset-dir', type=str, required=True, help='dataset directory')
    parser.add_argument('--image-type', type=str, default='default', help='Specification of input image type.')
    parser.add_argument('--n-max-test-iter', type=int, default=None, help='Max iteration on test.')
    parser.add_argument('--log-root-dir', type=str, default='./log/')
    parser.add_argument('--log-index', type=int, default=None)
    parser.add_argument('--output-root-dir', type=str, default=None)
    parser.add_argument('--stage-index', type=int, default=1)
    parser.add_argument('--store-codes', action='store_true', default=False)
    return parser

def load_train_args(param_dir):
    with open(os.path.join(param_dir, 'args.json')) as fp:
        return argparse.Namespace(**json.load(fp))

def str2bool(string):
    string = string.lower()
    if string in ('on', 'true', 'yes'):
        return True
    elif string in ('off', 'false', 'no'):
        return False
    else:
        raise ValueError('Unknown flag value: {}'.format(string))

def parse_index_file(filename):
    indices = []
    with open(filename) as fp:
        for line in fp.readlines():
            indices.append(line.strip())
    return indices


if __name__ == '__main__':
    main()
