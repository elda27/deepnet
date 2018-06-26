import sys
import argparse
import utils
from utils import mhd, visualizer, dataset
import process
import os
import os.path
import numpy as np
try:
    import cupy as cp
    xp = cp
    as_cpu_array = cp.asnumpy
    as_gpu_array = cp.array
except ImportError:
    xp = np
    as_cpu_array = np.asarray
    as_gpu_array = np.asarray
from chainer.iterators import MultiprocessIterator
import tqdm

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--truth-dir', type=str, required=True, help='Groundtruth dataset directory')
    parser.add_argument('--test-dir', type=str, nargs='+', required=True, help='Test prediciton result directory (accept multiple arguments)')
    parser.add_argument('--test-names', type=str, nargs='+', required=True, help='Case name of test prediction (This arugment is same length about --test-dir)')
    parser.add_argument('--test-index', type=str, nargs='+', required=True, help='Test index')
    parser.add_argument('--visualizer', type=str, nargs='+', required=True, help='Visualize images.')
    parser.add_argument('--output-dir', type=str, default='../output-metric')
    parser.add_argument('--output-filename')
    parser.add_argument('--num-visualization', type=int, default=100)
    parser.add_argument('--num-split-visualization', type=int, default=10)
    parser.add_argument('--shuffle-index', action='store_true', default=True)
    #parser.add_argument('--use-profile', action='store_true')

    if len(sys.argv) < 2:
        parser.print_help()

    args = parser.parse_args()
    
    if args.metrics is None:
        args.metrics = list(RegistedMetrics.keys())

    _main(args)

def _main(args):
    if args.gpu >= 0:
        #global xp
        xp = cp
    else:
        #global xp
        xp = np

    test_index_list = []
    for test_index = args.test_index
        if test_index == '*':
            test_index_list.append(test_index_list[-1])
        else:
            test_index_list.append(utils.parse_index_file(args.test_index))
    
    assert len(args.test_dir) % len(test_index_list) == 0, 'The number of test directory is multiply of the number of test_index list:{}'

    truth_dataset = dataset.XpDataset(args.truth_dir, test_index, image=True, label=True)
    pred_datasets = [dataset.XpDataset(td, test_index, image=False, label=True) for td in args.test_dir]

    truth_iter = MultiprocessIterator(truth_dataset, args.num_parallel, shuffle=False, repeat=False)
    pred_iters = [ MultiprocessIterator(pd, args.num_parallel, shuffle=False, repeat=False) for pd in pred_datasets ]

    nch_visualizer = visualizer.NchImageVisualizer(
        '{__iteration__:08d}.png', 10, 14,
        ['truth', ] + args.test_names,
        ['truth_overlap', ] + [ name + '_overlap' for name in args.test_names ],
        process.colors, subtract=[ ('truth', name) for name in args.test_names ]
    )

    all_metrics = {  }
    for i, batches in tqdm.tqdm(enumerate(zip(truth_iter, *pred_iters)), total = len(truth_dataset) // args.num_parallel):
        truth_vars = utils.batch_to_vars(batches[0])[0]
        pred_vars_list = [ utils.batch_to_vars(batch)[0] for batch in batches[1:] ]
    
        truth_vars['label'] = xp.concatenate( [ xp.expand_dims(xp.asarray(label), axis=0) for label in truth_vars['label'] ], axis=0)

        # For visualization
        variables = { '__iteration__': i }
        variables['truth'] = truth_vars['label']
        variables['truth_overlap'] = as_gpu_array(process.make_overlap_label(as_cpu_array(truth_vars['label']))[0])
        
        # Compute metrics
        thresholds = None
        for pred_vars, test_name in zip(pred_vars_list, args.test_names):
            pred_vars['label'] = xp.concatenate( [ xp.expand_dims(xp.asarray(label), axis=0) for label in pred_vars['label'] ], axis=0)
            values, thresholds = computeMetrics(truth_vars, pred_vars, args.metrics)
            #batch_metrics[name].extend(values)
            all_metrics = reform(all_metrics, test_name, values, thresholds)

            # For visualization
            if not nch_visualizer.is_last_finished:
                variables[test_name] = pred_vars['label']
                variables[test_name + '_overlap'] = as_gpu_array(process.make_overlap_label(as_cpu_array(pred_vars['label']))[0])

        nch_visualizer(variables)
        #all_metrics = reform(all_metrics, batch_metrics, thresholds)
    
    pd.DataFrame.from_dict(all_metrics).to_hdf('metrics.hdf5', 'metrics')

def reform(all_metrics, test_name, values, thresholds):
    for threshold, metric_list in zip(thresholds.tolist(), values):
        for metric_name, metric in metric_list.items():
            for rib_index in range(len(metric)):
                all_metrics.setdefault((test_name, metric_name, threshold, rib_index), []).extend(metric[:, rib_index].tolist())

    return all_metrics

RegistedMetrics = {}
def register_metric(name):
    def _register_metric(f):
        RegistedMetrics[name] = f
        return f
    return _register_metric

def computeMetrics(truth_vars, pred_vars, metrics, num_separate = 20):
    cases = truth_vars['case_name']

    thresholds = xp.linspace(0.0, 1.0, num_separate + 1)
    
    truth_labels = truth_vars['label']
    pred_labels = pred_vars['label']

    result_metric_list = []
    for thresh in thresholds:
        result_metrics = {}
        truth_bin_labels = truth_labels > 0
        pred_bin_labels = pred_labels > thresh
        for metric in metrics:
            result_metric = RegistedMetrics[metric](truth_bin_labels, pred_bin_labels, axes=tuple(i for i in range(2, truth_bin_labels.ndim)))
            result_metrics.update(result_metric)
        result_metric_list.append(result_metrics)
    return result_metric_list, thresholds

@register_metric('Dice')
def calcDice(im1, im2, axes):
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = xp.sum(im1, axis=axes) + xp.sum(im2, axis=axes)

    intersection = xp.logical_and(im1, im2)

    dice = 2.0 * xp.sum(intersection, axis=axes) / (im_sum + 1e-8) 
    return { 'Dice':dice }

@register_metric('Jaccard')
def calcJaccard(im1, im2, axes):
    im_or_sum  = xp.logical_or(im1, im2).sum(axis=axes)
    im_and_sum = xp.logical_and(im1, im2).sum(axis=axes)

    jaccard = im_and_sum.astype(xp.float32) / (im_or_sum + 1e-8)

    return { 'Jaccard':jaccard }

@register_metric('F_measure')
def calcStatisticalMetrics(bin_truth_label, bin_pred_label, axes):
    truth_positive = (bin_truth_label == 1)
    truth_negative = (bin_truth_label == 0)
    pred_positive = (bin_pred_label == 1)
    pred_negative = (bin_pred_label == 0)

    tp_indices = xp.logical_and(truth_positive, pred_positive)
    fn_indices = xp.logical_and(truth_positive, pred_negative)
    fp_indices = xp.logical_and(truth_negative, pred_positive)
    tn_indices = xp.logical_and(truth_negative, pred_negative)
    
    TP = tp_indices.sum(axis=axes)
    FN = fn_indices.sum(axis=axes)
    FP = fp_indices.sum(axis=axes)
    TN = tn_indices.sum(axis=axes)
    precesion = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    metrics = {
        'TPR': TP / (TP + FN + 1e-8),
        'FPR': FP / (FP + TN + 1e-8),
        'precesion': precesion,
        'recall':    recall,
        'F_measurement': 2* precesion * recall / (precesion + recall + 1e-8)
    }
    return metrics

if __name__ == '__main__':
    main()
