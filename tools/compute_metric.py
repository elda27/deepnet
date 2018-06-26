import auto_path

import argparse
from functools import reduce
from deepnet import utils
from deepnet.utils import mhd, visualizer, dataset
import process

xp = None
as_gpu_array = None
as_cpu_array = None
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

from chainer.iterators import MultiprocessIterator, SerialIterator
import tqdm
import pandas as pd
import xarray as xr
import numba

try:
    import line_profiler
    AVAILABLE_PROFILER=True
except:
    AVAILABLE_PROFILER=False

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--truth-dir', type=str, required=True, help='Groundtruth dataset directory')
    parser.add_argument('--test-dir', type=str, nargs='+', required=True, help='Test prediciton result directory (Ex. <Fold 1, Case 1 dir> <Fold1, Case 2 dir> <Fold2, Case 1 dir>  <Fold2, Case 2 dir>)')
    parser.add_argument('--test-names', type=str, nargs='+', required=True, help='Case name of test prediction (This arugment is same length about --test-dir)')
    parser.add_argument('--test-index', type=str, nargs='+', required=True, help='Test index')
    parser.add_argument('--output', type=str, default='metrics.nc')
    parser.add_argument('--num-parallel', type=int, default=5)
    parser.add_argument('--metrics', type=str, nargs='*', default=None)
    parser.add_argument('--use-profile', action='store_true')
    parser.add_argument('--n-image', type=int, default=None)

    if len(sys.argv) < 2:
        parser.print_help()

    args = parser.parse_args()

    if args.gpu >= 0:
        cp.cuda.Device(args.gpu).use()

    if args.metrics is None:
        args.metrics = list(RegistedMetrics.keys())

    pr = None
    if args.use_profile:
        assert AVAILABLE_PROFILER
        pr = line_profiler.LineProfiler()

        # Profiling about dataset
        for func in RegistedMetrics.values():
            pr.add_function(func)
        
        # Profiling about io and ip function
        pr.add_module(utils.dataset)

        # Profiling about trainer
        pr.add_function(_main)
        
        # Start profiling
        pr.enable()

    _main(args)

    if args.use_profile:
        pr.dump_stats('compute_metric.py.lprof')
        print('Finished to profile the time.\nTo show the result of the profile run this command.\n "python -m line_profiler compute_metric.py.lprof"')

def _main(args):
    global xp
    if args.gpu >= 0:
        xp = cp
    else:
        xp = np

    test_index_list = []
    for test_index in args.test_index:
        if test_index == '*':
            test_index_list.append(test_index_list[-1])
        else:
            test_index_list.append(utils.parse_index_file(test_index))
    
    assert len(args.test_dir) % len(test_index_list) == 0, 'The number of test directory is multiply of the number of test_index list:{}'

    all_metrics = []
    num_fold = len(args.test_dir) // len(args.test_names)
    for fold_index in tqdm.trange(num_fold):
        test_index = test_index_list[fold_index]
        test_dirs = args.test_dir[fold_index * len(args.test_names):(fold_index+1) * len(args.test_names)]

        all_metrics.append(compute_all_metrics(args, test_dirs, test_index))
        #for test_name, metrics in all_metrics_in_case.items():
        #    all_metrics[(fold_index, test_name)] = metrics
        #    #all_metrics.update({ (i, test_name, key): value for key, value in metrics.items() })
    
    metric_da = xr.concat(all_metrics, pd.Index(list(range(len(all_metrics))), name='fold'))

    output_dir = os.path.dirname(args.output)
    if output_dir is not None and output_dir != '':
        os.makedirs(output_dir, exist_ok=True)
    metric_da.to_netcdf(args.output)


def compute_all_metrics(args, test_dir, test_index):
    truth_dataset = dataset.XpDataset(args.truth_dir, test_index, image=False, label=True)
    pred_datasets = [dataset.XpDataset(td, test_index, image=False, label=True) for td in test_dir]

    truth_iter = MultiprocessIterator(truth_dataset, args.num_parallel, shuffle=False, repeat=False)
    pred_iters = [ MultiprocessIterator(pd, args.num_parallel, shuffle=False, repeat=False) for pd in pred_datasets ]
    #truth_iter = SerialIterator(truth_dataset, args.num_parallel, shuffle=False, repeat=False)
    #pred_iters = [ SerialIterator(pd, args.num_parallel, shuffle=False, repeat=False) for pd in pred_datasets ]

    #all_metrics = {  }
    batch_results_dict = {}
    for i, batches in tqdm.tqdm(enumerate(zip(truth_iter, *pred_iters)), total = len(truth_dataset) // args.num_parallel):
        if args.n_image is not None and args.n_image <= i:
            break

        truth_vars = utils.batch_to_vars(batches[0])[0]
        pred_vars_list = [ utils.batch_to_vars(batch)[0] for batch in batches[1:] ]
    
        truth_vars['label'] = xp.concatenate( [ xp.expand_dims(xp.asarray(label), axis=0) for label in truth_vars['label'] ], axis=0)

        # Compute metrics
        for pred_vars, test_name in zip(pred_vars_list, args.test_names):
            pred_vars['label'] = xp.concatenate( [ xp.expand_dims(xp.asarray(label), axis=0) for label in pred_vars['label'] ], axis=0)
            values = computeMetrics(truth_vars, pred_vars, args.metrics)
            batch_results_dict.setdefault(test_name,[]).append(values)
            #all_metrics.setdefault(test_name, dict())
    
    new_axis = list(batch_results_dict.keys())
    new_data = []
    for test_name, batch_results in batch_results_dict.items():
        data_dict = {}
        for batch_result in batch_results:
            for key, values in batch_result.items():
                data_dict.setdefault(key, []).append(values)
        new_data.append(data_dict)
    
    new_data = [ xr.Dataset({ key: xr.concat(value, dim='case_name') for key, value in dataset_.items() }) for dataset_ in new_data ]

    return xr.concat(new_data, pd.Index(new_axis, name='test_name'))

RegistedMetrics = {}
RegisterKeys = {}
def register_metric(name, keys=None):
    def _register_metric(f):
        RegistedMetrics[name] = f
        RegisterKeys[name] = [ name ] if keys is None else keys 
        return f
    return _register_metric

#@numba.jit
def computeMetrics(truth_vars, pred_vars, metrics, num_separate = 20):
    cases = truth_vars['case_name']

    thresholds = (xp.arange(num_separate + 1) / num_separate).astype(xp.float32)
    
    truth_labels = truth_vars['label']
    pred_labels = pred_vars['label']

    # (num thresh, cases, num rib bones)
    shape = (
        len(thresholds), 
        truth_labels.shape[0],
        truth_labels.shape[1], 
    )

    result_metrics = {}
    for i, thresh in enumerate(thresholds):
        truth_bin_labels = truth_labels > 0
        pred_bin_labels = pred_labels > thresh
        for metric_name in metrics:
            _metrics = RegistedMetrics[metric_name](truth_bin_labels, pred_bin_labels, axes=tuple(i for i in range(2, truth_bin_labels.ndim)))
            for key, values in _metrics.items():
                if key not in result_metrics:
                    result_metrics[key] = xp.zeros(shape)
                result_metrics[key][i] = values

    dims = ['threshold', 'case_name', 'rib_index']
    coords = {
        'threshold': [float('{:.3f}'.format(float(as_cpu_array(t)))) for t in thresholds],
        'case_name': cases,
        'rib_index': list(range(shape[2])),
    }
    return { key: xr.DataArray(as_cpu_array(values), dims=dims, coords=coords) for key, values in result_metrics.items() }

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

@register_metric('F_measure', keys=[
    'TPR', 
    'FPR', 
    'precesion', 
    'recall', 
    'F_measurement', 
])
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
