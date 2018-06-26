import argparse
import utils
from utils.network import NetworkNode
import numpy as np
try:
    import cupy as cp
except ImportError:
    pass
import chainer
import chainer.functions as F
from chainer import cuda
import concurrent.futures
import network
import datetime
import glob
import math
import os
import os.path
from functools import reduce
from itertools import cycle
import json

import connections

def main():
    parser = build_arguments()
    args = parser.parse_args()

    assert args.stage_index > 0

    train_index = parse_index_file(args.train_index)
    valid_index = parse_index_file(args.valid_index)
    print('Train index', train_index)
    print('Valid index', valid_index)
    #train_dataset = utils.dataset.InstantDataset(args.dataset_dir, use_ratio=0.8, use_backward=False)
    #valid_dataset = utils.dataset.InstantDataset(args.dataset_dir, use_ratio=0.2, use_backward=True)
    train_dataset = utils.dataset.XpDataset(args.dataset_dir, train_index, args.image_type, image=(args.stage_index >= 2))
    valid_dataset = utils.dataset.XpDataset(args.dataset_dir, valid_index, args.image_type, image=(args.stage_index >= 2))
    
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

    optimizer = chainer.optimizers.Adam(args.lr_rate)

    # Setup other environment for each stage
    logger = []
    archive_nodes = []
    optimizing_loss = ''

    archive_nodes = ['CAE']
    optimizing_loss = 'loss_reconstruct'
    logger.append(
        utils.logger.CsvLogger(os.path.join(log_dir, 'log_stage1.csv'),
            [
                '__train_iteration__', 
                'train.loss_reconstruct', 'valid.loss_reconstruct',
            ])
    )

    # models
    cae_model = network.conv_auto_encoder.ConvolutionalAutoEncoder(
        2, args.n_channel, encode_dim=args.encode_dims, 
        n_layers=args.n_layers,
        dropout=args.dropout_mode, use_batch_norm=args.use_batch_norm,
        )
        
    optimizer.setup(cae_model)    
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.01))

    # Costruct network model
    network_manager, visualizers = build_network(cae_model, args, log_dirs)

    # Save variables
    with open(os.path.join(param_dir, 'args.json'), 'w+') as fp:
        json.dump(vars(args), fp, indent=2)
    
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
    train_config['progress_vars'] = ['{}:.3f'.format(optimizing_loss)]

    trainer = utils.trainer.Trainer(
        network=network_manager,
        train_iter=chainer.iterators.MultiprocessIterator(train_dataset, args.batch_size, shuffle=True, repeat=True),
        valid_iter=chainer.iterators.MultiprocessIterator(valid_dataset, args.batch_size, shuffle=False, repeat=False),
        visualizers=visualizers,
        optimizer={optimizing_loss: optimizer},
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
    parser.add_argument('--n-dim', type=int, default=2, help='n dimension of input data')
    parser.add_argument('--n-channel', type=int, default=14, help='n channel of input data')
    parser.add_argument('--n-layers', type=int, default=5, help='n channel of input data')
    parser.add_argument('--train-index', type=str, required=True, help='training indices text')
    parser.add_argument('--valid-index', type=str, required=True, help='validation indices text')
    parser.add_argument('--lr-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n-max-train-iter', type=int, default=60000, help='Max iteration of train.')
    parser.add_argument('--n-max-valid-iter', type=int, default=None, help='Max iteration of validation.')
    parser.add_argument('--n-valid-step', type=int, default=5000, help='Step of validation every this iteration.')
    parser.add_argument('--dataset-dir', type=str, required=True, help='dataset directory')
    parser.add_argument('--log-root-dir', type=str, default='./log/')
    parser.add_argument('--dropout-mode', type=str, default='dropout')
    parser.add_argument('--use-batch-norm', type=str2bool, default=True)
    parser.add_argument('--denoisy', type=str2bool, default=True)
    parser.add_argument('--encode-dims', type=int, default=64)
    parser.add_argument('--log-index', type=int, default=0, help='Log direcotry index for training.')

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

def parse_index_file(filename):
    indices = []
    with open(filename) as fp:
        for line in fp.readlines():
            indices.append(line.strip())
    return indices

import process as p

def build_network(cae_model, args, log_dirs):
    network_manager = utils.network.NetworkManager(['image'])
    # Common difinition
    if args.gpu < 0:
        network_manager.add('copy', NetworkNode('image', 'gpu_image', p.to_cpu))
    else:
        network_manager.add('to_gpu', NetworkNode('image', 'gpu_image', p.to_gpu))
        cae_model.to_gpu()

    if args.denoisy:
        network_manager.add('noisy', NetworkNode('gpu_image', 'gpu_noisy_image', p.apply_gaussian_noise, sigma=10, device=args.gpu))
    else:
        network_manager.add('noisy', NetworkNode('gpu_image', 'gpu_noisy_image', F.copy, dst=args.gpu))

    network_manager.add('CAE', NetworkNode('gpu_noisy_image', 'gpu_reconstruct_image', cae_model, updatable=True))

    # loss definition
    network_manager.add('loss_euclidean', NetworkNode(['gpu_reconstruct_image', 'gpu_image'], 'loss_euclidean', p.loss.euclidean_distance, test=False))
    #network_manager.add('loss_cae', NetworkNode(['loss_sigmoid', 'loss_softmax', 'loss_euclidean'], 'loss_reconstruct', lambda *xs: sum(xs), test=False))

    # For visualization
    network_manager.add('make_overlap_label', NetworkNode(
        ['gpu_label', 'gpu_noisy_label', 'gpu_reconstruct_label'], 
        ['gpu_overlap_label', 'gpu_noisy_overlap_label', 'gpu_overlap_reconstruct_label'], 
        p.make_overlap_label, training=False, validation=True, test=False))

    # Generate visualizers
    visualize_dir = log_dirs['visualize']
    tile_img_filename = os.path.join(visualize_dir, '{__train_iteration__:08d}_tile.png')
    tile_visualizer = utils.visualizer.TileImageVisualizer(tile_img_filename, (5, 5), ['gpu_overlap_label', 'gpu_noisy_overlap_label', 'gpu_overlap_reconstruct_label'], (1, 3))
    #tile_visualizer = utils.visualizer.TileImageVisualizer(tile_img_filename, 25, (5, 5), ['label', 'gpu_noisy_label', 'gpu_reconstruct_label'], (1, 3))

    n_ch_tile_img_filename = os.path.join(visualize_dir, '{__train_iteration__:08d}_nch_tile.png')
    n_ch_visualizer = utils.visualizer.NchImageVisualizer(
        n_ch_tile_img_filename, 5, args.n_channel, 
        ['label', 'gpu_noisy_label', 'gpu_reconstruct_label'], 
        ['gpu_overlap_label', 'gpu_noisy_overlap_label', 'gpu_overlap_reconstruct_label'], 
        color_pallete=p.colors, subtract=[('label', 'gpu_reconstruct_label')]
        )

    label_img_filename = os.path.join(visualize_dir, '{__train_iteration__:08d}_{__name__}_{__index__:03d}.mhd')
    mhd_writer = utils.visualizer.MhdImageWriter(
        label_img_filename, 3, 
        ['label', 'gpu_noisy_label', 'gpu_reconstruct_label']
        )

    #architecture_filename = os.path.join(visualize_dir, '{__name__}.dot')
    #network_architecture_writer = utils.visualizer.NetworkArchitectureVisualizer(architecture_filename, 'loss_reconstruct') 

    return network_manager, [tile_visualizer, mhd_writer, n_ch_visualizer]


if __name__ == '__main__':
    main()
