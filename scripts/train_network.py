import argparse
from deepnet.utils.network import NetworkNode
from deepnet import *
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

import connections

def main():
    parser = build_arguments()
    args = parser.parse_args()

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()

    assert args.stage_index > 0

    train_index = parse_index_file(args.train_index)
    valid_index = parse_index_file(args.valid_index)

    train_dataset = utils.dataset.GeneralDataset(args.dataset_config, train_index)
    valid_dataset = utils.dataset.GeneralDataset(args.dataset_config, valid_index)
    
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
    if args.stage_index == 1:
        archive_nodes = ['CAE']
        optimizing_loss = 'loss_reconstruct'
        logger.append(
            utils.logger.CsvLogger(os.path.join(log_dir, 'log_stage1.csv'),
                [
                    '__train_iteration__', 
                    'train.loss_softmax', 'valid.loss_softmax',
                    'train.loss_sigmoid', 'valid.loss_sigmoid',
                    'train.loss_euclidean', 'valid.loss_euclidean',
                    'train.loss_reconstruct', 'valid.loss_reconstruct',
                ])
        )
    elif args.stage_index == 2:
        archive_nodes = ['Segnet']
        optimizing_loss = 'loss_segment'
        logger.append(
            utils.logger.CsvLogger(os.path.join(log_dir, 'log_stage2.csv'),
                [
                    '__train_iteration__', 
                    'train.loss_segment_softmax', 'valid.loss_segment_softmax',
                    'train.loss_segment_sigmoid', 'valid.loss_segment_sigmoid',
                    #'train.loss_segment_euclidean', 'valid.loss_segment_euclidean',
                    'train.loss_segment', 'valid.loss_segment',
                ])
        )
    elif args.stage_index == 3:
        archive_nodes = ['Segnet', 'GroundtruthEncoder']
        optimizing_loss = 'loss_total'
        logger.append(
            utils.logger.CsvLogger(os.path.join(log_dir, 'log_stage3.csv'),
                [
                    '__train_iteration__', 
                    #'train.loss_segment_softmax', 'valid.loss_segment_softmax',
                    'train.loss_segment_sigmoid', 'valid.loss_segment_sigmoid',
                    'train.loss_encode_dims', 'valid.loss_encode_dims',
                    'train.loss_total', 'valid.loss_total',
                ])
        )

    # models
    cae_model = network.conv_auto_encoder.ConvolutionalAutoEncoder(
        2, args.n_channel, encode_dim=args.encode_dims, 
        n_layers=args.n_layers,
        dropout=args.dropout_mode, use_batch_norm=args.use_batch_norm,
        )
    segnet_model = network.tl_net.Segnet(
        2, 1, cae_model.decoder,
        use_skipping_connection=args.use_skipping_connection if args.stage_index >= 3 else 'none'
    )
    
    if args.stage_index == 1:
        optimizer.setup(cae_model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.01))

    elif args.stage_index == 2:
        optimizer.setup(segnet_model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))
        
        model_archive = list(glob.glob(os.path.join(log_dir, 'model_stage' + str(args.stage_index - 1), 'CAE_*.npz')))[-1]
        chainer.serializers.load_npz(model_archive, cae_model)

        segnet_model.decoder.disable_update()
    elif args.stage_index == 3:
        optimizer.setup(cae_model)
        optimizer.setup(segnet_model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))
        
        cae_archive = list(glob.glob(os.path.join(log_dir, 'model_stage1', 'CAE_*.npz')))[-1]
        chainer.serializers.load_npz(cae_archive, cae_model)
        
        segnet_archive = list(glob.glob(os.path.join(log_dir, 'model_stage2', 'Segnet_*.npz')))[-1]
        chainer.serializers.load_npz(segnet_archive, segnet_model)
        

    # Costruct network model
    network_manager, visualizers = connections.build_network_for_ribcage(cae_model, segnet_model, args, log_dirs, args.stage_index)
    #network_manager, visualizers = build_network_for_real_image(cae_model, args, log_dir)

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
    
    parser.add_argument('--n-channel', type=int, default=14, help='n channel of input data')
    parser.add_argument('--n-layers', type=int, default=5, help='n channel of input data')

    parser.add_argument('--train-index', type=str, required=True, help='training indices text')
    parser.add_argument('--valid-index', type=str, required=True, help='validation indices text')
    parser.add_argument('--image-type', type=str, default='default', help='Specification of input image type.')
    parser.add_argument('--lr-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n-max-train-iter', type=int, default=60000, help='Max iteration of train.')
    parser.add_argument('--n-max-valid-iter', type=int, default=None, help='Max iteration of validation.')
    parser.add_argument('--n-valid-step', type=int, default=5000, help='Step of validation every this iteration.')

    parser.add_argument('--dataset-dir', type=str, required=True, help='dataset directory')
    parser.add_argument('--log-root-dir', type=str, default='./log/')
    parser.add_argument('--use-skipping-connection', type=str, default='none', help='Skipping connection method(none[default], add)')
    parser.add_argument('--dropout-mode', type=str, default='dropout')
    parser.add_argument('--use-batch-norm', type=str2bool, default=True)
    parser.add_argument('--denoisy', type=str2bool, default=True)
    parser.add_argument('--encode-dims', type=int, default=64)
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

def parse_index_file(filename):
    indices = []
    with open(filename) as fp:
        for line in fp.readlines():
            indices.append(line.strip())
    return indices

if __name__ == '__main__':
    main()
