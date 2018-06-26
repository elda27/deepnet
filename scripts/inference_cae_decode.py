import deepnet.network
from deepnet import utils
from deepnet import process
#import deepnet.network.conv_auto_encoder as cae
import chainer
from chainer.serializers import load_npz
import chainer.functions as F

import argparse
import os
import os.path
import glob
import numpy as np
import json
import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for inference.')
    parser.add_argument('--log-root-dir', type=str, default='./log/')
    parser.add_argument('--log-index', type=int, default=None)
    parser.add_argument('--output-root-dir', type=str, default=None)
    parser.add_argument('--axes', type=int, nargs=2, default=(0, 1), help='Visualize to decode representaiton variable on axis.')
    parser.add_argument('--n-visualize', type=int, default=10)
    parser.add_argument('--shape', type=int, nargs=2, default=(336, 336))
    args = parser.parse_args()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    log_dir = utils.get_log_dir(args.log_root_dir, args.log_index)
    if args.output_root_dir is None:
        args.output_root_dir = os.path.join(log_dir, 'test_decode_stage1')
    
    train_args = load_train_args(os.path.join(log_dir, 'param_stage1'))

    cae_model = deepnet.network.conv_auto_encoder.ConvolutionalAutoEncoder(
            2, train_args.n_channel, encode_dim=train_args.encode_dims, 
            n_layers=train_args.n_layers,
            dropout=train_args.dropout_mode, use_batch_norm=train_args.use_batch_norm,
            use_skipping_connection = 'none',
        )

    cae_archives = list(glob.glob(os.path.join(log_dir, 'model_stage1', 'CAE_*.npz')))
    assert len(cae_archives) != 0, 'Trained model is not found at {}'.format(os.path.join(log_dir, 'model_stage1'))
    load_npz(cae_archives[-1], cae_model)

    if args.gpu >= 0:
        cae_model.to_gpu()

    first_axis, second_axis = args.axes
    first_values = np.linspace(-1.0, 1.0, args.n_visualize)
    second_values = np.linspace(-1.0, 1.0, args.n_visualize)

    images = []
    os.makedirs(args.output_root_dir, exist_ok=True)
    tile_visualizer = utils.visualizer.TileImageVisualizer(os.path.join(args.output_root_dir, 'tile.png'), (args.n_visualize, args.n_visualize), ['overlap_label'], (1,1))
    #image_writer = utils.visualizer.MhdImageWriter(os.path.join(args.output_root_dir, '{__index__}_label.mhd'), args.n_visualize ** 2, ['label'])

    for i1 in tqdm.trange(args.n_visualize):
        v1 = first_values[i1]
        for i2 in tqdm.trange(args.n_visualize // args.batch_size):
            v2 = second_values[i2 * args.batch_size : (i2 + 1) * args.batch_size]
            array = np.zeros((args.batch_size, cae_model.decoder.input_dim), dtype=np.float32)
            array[:, first_axis] = v1
            array[:, second_axis] = v2

            if args.gpu >= 0:
                array = chainer.Variable(array)
                array.to_gpu()

            reconstruct_image = cae_model.decoder(array)
            reconstruct_image = deepnet.network.conv_auto_encoder.utils.crop(reconstruct_image, reconstruct_image.shape[0:2] + args.shape)
            reconstruct_image = F.sigmoid(reconstruct_image)

            reconstruct_overlap_image = process.make_overlap_label(reconstruct_image)[0]
            images = { 'label': reconstruct_image, 'overlap_label':reconstruct_overlap_image, '__iteration__': i1 * args.n_visualize // args.batch_size + i2 }
            tile_visualizer(images)
            #image_writer(images)
    tile_visualizer.save()
    #image_writer.save()

def load_train_args(param_dir):
    with open(os.path.join(param_dir, 'args.json')) as fp:
        return argparse.Namespace(**json.load(fp))

if __name__ == '__main__':
    main()
