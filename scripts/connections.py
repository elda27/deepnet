
import os.path
from deepnet import utils
from deepnet.utils.network import NetworkNode
import deepnet.process as p

import chainer
import chainer.functions as F
import numpy as np
try:
    import cupy as cp
except ImportError:
    pass
import functools

def build_network_for_ribcage(cae_model, segnet_model, args, log_dirs, stage_index):
    p.set_gpu_id(args.gpu)
    if stage_index == 1:
        return build_network_for_ribcage_on_stage1(cae_model, args, log_dirs)
    elif stage_index == 2:
        return build_network_for_ribcage_on_stage2(segnet_model, args, log_dirs)
    elif stage_index == 3:
        return build_network_for_ribcage_on_stage3(cae_model, segnet_model, args, log_dirs)

def build_network_for_ribcage_on_stage1(cae_model, args, log_dirs):
    network_manager = utils.network.NetworkManager(['label'])
    # Common difinition
    if args.gpu < 0:
        network_manager.add('copy', NetworkNode('label', 'gpu_label', p.to_cpu))
    else:
        network_manager.add('to_gpu', NetworkNode('label', 'gpu_label', p.to_gpu))
        cae_model.to_gpu()

    if args.denoisy:
        network_manager.add('noisy', NetworkNode('gpu_label', 'gpu_noisy_label', p.apply_gaussian_noise, clip=(0.0, 1.0), sigma=0.1, device=args.gpu))
    else:
        network_manager.add('noisy', NetworkNode('gpu_label', 'gpu_noisy_label', F.copy, dst=args.gpu))

    # Biasいるかな？
    #network_manager.add('bias', NetworkNode('gpu_noisy_label', 'gpu_noisy_label_biased', p.bias, multiply=100.0))
    #network_manager.add('CAE', NetworkNode('gpu_noisy_label_biased', 'gpu_raw_reconstruct_label', cae_model, updatable=True))
    network_manager.add('CAE', NetworkNode('gpu_noisy_label', 'gpu_raw_reconstruct_label', cae_model, updatable=True))

    # loss definition
    network_manager.add('sigmoid', NetworkNode('gpu_raw_reconstruct_label', 'gpu_reconstruct_label', F.sigmoid, training=True, validation=True, test=True))
    network_manager.add('cast_label', NetworkNode('gpu_label', 'i_gpu_label', p.cast_type, dtype=cp.int32, test=False))
    network_manager.add('loss_sigmoid', NetworkNode(['gpu_raw_reconstruct_label', 'i_gpu_label'], 'loss_sigmoid', F.sigmoid_cross_entropy, normalize=False, test=False))
    network_manager.add('loss_softmax', NetworkNode(['gpu_reconstruct_label', 'i_gpu_label'], 'loss_softmax', p.loss.total_softmax_cross_entropy, normalize=False, test=False))
    network_manager.add('loss_euclidean', NetworkNode(['gpu_reconstruct_label', 'gpu_label'], 'loss_euclidean', p.loss.euclidean_distance, test=False))
    network_manager.add('loss_cae', NetworkNode(['loss_sigmoid', 'loss_softmax', 'loss_euclidean'], 'loss_reconstruct', lambda *xs: sum(xs), test=False))

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

def build_network_for_ribcage_on_stage2(segnet_model, args, log_dirs):
    network_manager = utils.network.NetworkManager(['image', 'label'])
    # Common difinition
    if args.gpu < 0:
        network_manager.add('copy_label', NetworkNode('label', 'gpu_label', p.to_cpu))
        network_manager.add('copy_image', NetworkNode('image', 'gpu_image', p.to_cpu))
    else:
        network_manager.add('copy_label', NetworkNode('label', 'gpu_label', p.to_gpu))
        network_manager.add('copy_image', NetworkNode('image', 'gpu_image', p.to_gpu))
        segnet_model.to_gpu()

    network_manager.add('Segnet', NetworkNode('gpu_image', 'gpu_raw_segment_label', segnet_model, updatable=True))

    # loss definition
    network_manager.add('sigmoid_segment_label', NetworkNode('gpu_raw_segment_label', 'gpu_segment_label', F.sigmoid, training=True, validation=True, test=True))
    network_manager.add('cast_label', NetworkNode('gpu_label', 'i_gpu_label', p.cast_type, dtype=cp.int32, test=False))
    network_manager.add('loss_segment_sigmoid', NetworkNode(['gpu_raw_segment_label', 'i_gpu_label'], 'loss_segment_sigmoid', F.sigmoid_cross_entropy, normalize=False, test=False))
    network_manager.add('loss_segment_softmax', NetworkNode(['gpu_segment_label', 'i_gpu_label'], 'loss_segment_softmax', p.loss.total_softmax_cross_entropy, normalize=False, test=False))
    #network_manager.add('loss_euclidean', NetworkNode(['gpu_segment_label', 'gpu_label'], 'loss_euclidean', p.loss.euclidean_distance, test=False))
    network_manager.add('loss_segment', NetworkNode(['loss_segment_sigmoid', 'loss_segment_softmax'], 'loss_segment', lambda *xs: sum(xs), test=False))

    # For visualization
    network_manager.add('make_segnet_overlap_label', NetworkNode(
        ['gpu_label', 'gpu_segment_label'], 
        ['gpu_overlap_label', 'gpu_overlap_segment_label'], 
        p.make_overlap_label, training=False, validation=True, test=False))

    network_manager.add('transpose_image', 
        NetworkNode('gpu_image', 'gpu_transpose_image', F.transpose, axes=[0, 1, 3, 2], training=False, validation=True, test=False)
        )
    network_manager.add('blend_segnet_label_real_image', NetworkNode(
        ['gpu_overlap_label', 'gpu_transpose_image', 'gpu_overlap_segment_label', 'gpu_transpose_image'],
        ['gpu_blend_label_image', 'gpu_blend_segmentlabel_image'],
        p.blend_image, 
        training=False, validation=True, test=False
    ))
    
    # Generate visualizers
    visualize_dir = log_dirs['visualize']
    tile_img_filename = os.path.join(visualize_dir, '{__train_iteration__:08d}_tile.png')
    tile_visualizer = utils.visualizer.TileImageVisualizer(
        tile_img_filename, (5, 5), 
        [
            'gpu_transpose_image', 'gpu_overlap_label', 'gpu_overlap_segment_label',
            'gpu_transpose_image', 'gpu_blend_label_image', 'gpu_blend_segmentlabel_image',
        ], (2, 3)
        )

    n_ch_tile_img_filename = os.path.join(visualize_dir, '{__train_iteration__:08d}_nch_tile.png')
    n_ch_visualizer = utils.visualizer.NchImageVisualizer(
        n_ch_tile_img_filename, 5, args.n_channel, 
        ['label', 'gpu_segment_label'], 
        ['gpu_blend_label_image', 'gpu_blend_segmentlabel_image'], 
        color_pallete=p.colors, subtract=[('label', 'gpu_segment_label')]
        )

    label_img_filename = os.path.join(visualize_dir, '{__train_iteration__:08d}_{__name__}_{__index__:03d}.mhd')
    mhd_writer = utils.visualizer.MhdImageWriter(label_img_filename, 3, ['image', 'label', 'gpu_segment_label'])

    #architecture_filename = os.path.join(visualize_dir, '{__name__}.dot')
    #network_architecture_writer = utils.visualizer.NetworkArchitectureVisualizer(architecture_filename, 'loss_reconstruct') 

    return network_manager, [tile_visualizer, mhd_writer, n_ch_visualizer]

def build_network_for_ribcage_on_stage3(cae_model, segnet_model, args, log_dirs):
    network_manager = utils.network.NetworkManager(['image', 'label'])
    # Common difinition
    xp = None
    if args.gpu < 0:
        network_manager.add('copy_label', NetworkNode('label', 'gpu_label', p.to_cpu))
        network_manager.add('copy_image', NetworkNode('image', 'gpu_image', p.to_cpu))
        xp = np
    else:
        network_manager.add('copy_label', NetworkNode('label', 'gpu_label', p.to_gpu))
        network_manager.add('copy_image', NetworkNode('image', 'gpu_image', p.to_gpu))
        segnet_model.to_gpu()
        cae_model.encoder.to_gpu()
        xp = cp

    network_manager.add('Segnet', NetworkNode('gpu_image', 'gpu_raw_segment_label', segnet_model, updatable=True))
    network_manager.add('sigmoid_segment_label', NetworkNode('gpu_raw_segment_label', 'gpu_segment_label', F.sigmoid, training=True, validation=True, test=True))
    network_manager.add('GroundtruthEncoder', NetworkNode('gpu_label', 'groundtruth_encode_dims', cae_model.encoder, updatable=True))
    network_manager.add('PredictionEncoder', NetworkNode('gpu_segment_label', 'prediction_encode_dims', cae_model.encoder, updatable=True))

    # loss definition
    #loss_weight = [1.0, 1.0, 1e6]
    #def weight_sum(*xs):
    #    src_iter = zip(xs, loss_weight)
    #    x, w = next(src_iter)
    #    output = x * w 
    #    for x, w in src_iter:
    #        output += x * w
    #    return output

    network_manager.add('cast_label', NetworkNode('gpu_label', 'i_gpu_label', p.cast_type, dtype=cp.int32, test=False))
    network_manager.add('loss_segment_sigmoid', NetworkNode(['gpu_raw_segment_label', 'i_gpu_label'], 'loss_segment_sigmoid', F.sigmoid_cross_entropy, normalize=False, test=False))
    #network_manager.add('loss_segment_softmax', NetworkNode(['gpu_segment_label', 'i_gpu_label'], 'loss_segment_softmax', p.loss.total_softmax_cross_entropy, normalize=False, test=False))
    network_manager.add('loss_encode_dims_raw', NetworkNode(['groundtruth_encode_dims', 'prediction_encode_dims'], 'loss_encode_dims_raw', p.loss.euclidean_distance, test=False))
    network_manager.add('loss_encode_dims', NetworkNode('loss_encode_dims_raw', 'loss_encode_dims', p.bias, multiply=1e6, test=False))
    #network_manager.add('loss_total', NetworkNode(['loss_segment_sigmoid', 'loss_segment_softmax', 'loss_encode_dims'], 'loss_total', weight_sum, test=False))
    #network_manager.add('loss_total', NetworkNode(['loss_segment_sigmoid', 'loss_segment_softmax', 'loss_encode_dims'], 'loss_total', lambda *xs: sum(xs), test=False))
    network_manager.add('loss_total', NetworkNode(['loss_segment_sigmoid', 'loss_encode_dims'], 'loss_total', lambda *xs: sum(xs), test=False))

    # For visualization
    network_manager.add('make_segnet_overlap_label', NetworkNode(
        ['gpu_label', 'gpu_segment_label'], 
        ['gpu_overlap_label', 'gpu_overlap_segment_label'], 
        p.make_overlap_label, training=False, validation=True, test=False))

    network_manager.add('transpose_image', 
        NetworkNode('gpu_image', 'gpu_transpose_image', F.transpose, axes=[0, 1, 3, 2], training=False, validation=True, test=False)
        )
    network_manager.add('blend_segnet_label_real_image', NetworkNode(
        ['gpu_overlap_label', 'gpu_transpose_image', 'gpu_overlap_segment_label', 'gpu_transpose_image'],
        ['gpu_blend_label_image', 'gpu_blend_segmentlabel_image'],
        p.blend_image, 
        training=False, validation=True, test=False
    ))
    
    # Generate visualizers
    visualize_dir = log_dirs['visualize']
    tile_img_filename = os.path.join(visualize_dir, '{__train_iteration__:08d}_tile.png')
    tile_visualizer = utils.visualizer.TileImageVisualizer(
        tile_img_filename, (5, 5), 
        [
            'gpu_transpose_image', 'gpu_overlap_label', 'gpu_overlap_segment_label',
            'gpu_transpose_image', 'gpu_blend_label_image', 'gpu_blend_segmentlabel_image',
        ], (2, 3)
        )

    n_ch_tile_img_filename = os.path.join(visualize_dir, '{__train_iteration__:08d}_nch_tile.png')
    n_ch_visualizer = utils.visualizer.NchImageVisualizer(
        n_ch_tile_img_filename, 5, args.n_channel, 
        ['label', 'gpu_segment_label'], 
        ['gpu_blend_label_image', 'gpu_blend_segmentlabel_image'], 
        color_pallete=p.colors, subtract=[('label', 'gpu_segment_label')]
        )

    label_img_filename = os.path.join(visualize_dir, '{__train_iteration__:08d}_{__name__}_{__index__:03d}.mhd')
    mhd_writer = utils.visualizer.MhdImageWriter(label_img_filename, 3, ['image', 'label', 'gpu_segment_label'])

    return network_manager, [tile_visualizer, mhd_writer, n_ch_visualizer]

def build_network_for_real_image(cae_model, args, log_dir):
    network_manager = utils.network.NetworkManager(['label'])
    
    if args.gpu < 0:
        network_manager.add('copy', NetworkNode('label', 'gpu_label', p.to_cpu))
    else:
        network_manager.add('to_gpu', NetworkNode('label', 'gpu_label', p.to_gpu))
        cae_model.to_gpu()

    if args.denoisy:
        network_manager.add('noisy', NetworkNode('gpu_label', 'gpu_noisy_label', p.apply_gaussian_noise, clip=(0.0, 1.0), sigma=0.05, device=args.gpu))
    else:
        network_manager.add('noisy', NetworkNode('gpu_label', 'gpu_noisy_label', F.copy, dst=args.gpu))
    network_manager.add('CAE', NetworkNode('gpu_noisy_label', 'gpu_raw_reconstruct_label', cae_model, updatable=True))

    # loss_definition
    network_manager.add('loss_cae', NetworkNode(['gpu_label', 'gpu_raw_reconstruct_label'], 'loss_reconstruct', p.loss.euclidean_distance))


    # Generate visualizers
    visualize_dir = os.path.join(log_dir, 'visualize')
    tile_img_filename = os.path.join(visualize_dir, '{__train_iteration__:08d}_tile.png')
    tile_visualizer = utils.visualizer.TileImageVisualizer(tile_img_filename, (5, 5), ['label', 'gpu_noisy_label', 'gpu_raw_reconstruct_label'], (1, 3))

    return network_manager, [ tile_visualizer ]
