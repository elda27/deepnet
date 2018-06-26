import sys
import argparse
import os
import os.path

import glob
import random
import re

import mhd

import numpy as np
import pandas as pd
import xarray as xr
import numba
import tqdm
import itertools

import seaborn as sns
import matplotlib.pyplot as plt

plot_table = {}
def addplot(name):
    def _addplot(f):
        plot_table[name] = f
        return f
    return _addplot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--metric', nargs='+', type=str, required=True, help="The metric filename to be loaded.")
    parser.add_argument('-l', '--label', nargs='+', type=str, required=False, default=None, help="The metric filename to be loaded.")
    parser.add_argument('-p', '--plot', nargs='+', type=str, required=True, help='Plot method.({})'.format(', '.join(list(plot_table.keys()))))
    parser.add_argument('-a', '--axes', nargs='+', type=str, required=True, help='Chart axes separated hyphenation.(ex. y-x, y-x-c)')
    parser.add_argument('-al', '--axes-label', nargs='*', type=str, required=False, help='Chart axes text separated hyphenation.(ex. Y-X)')
    parser.add_argument('-op', '--output-prefix', nargs='+', type=str, required=True, help='The filename prefix to be saved.')
    parser.add_argument('-f', '--format', nargs='*', type=str, default=['png'], help='The file extension without dot (.) to be saved.')
    parser.add_argument('-q', '--query', nargs='*', type=str, default=[], help='The metric choose by the query.')
    parser.add_argument('--xlim', nargs=2, type=float, default=None, help='Set the x limits.')
    parser.add_argument('--ylim', nargs=2, type=float, default=None, help='Set the y limits.')
    parser.add_argument('--x-log', type=bool, default=False, help='Set x-axis to log scale.')
    parser.add_argument('--y-log', type=bool, default=False, help='Set y-axis to log scale.')

    if len(sys.argv) < 2:
        parser.print_help()
        return


    args = parser.parse_args()
    
    assert args.label is None or len(args.label) == len(args.metric)
    metric = None
    if len(args.metric) > 1:
        metric = None
        all_metrics = [ xr.open_dataset(m) for m in args.metric]
        if args.label is None:
            metric = xr.concat(all_metrics, pd.Index(args.label, name='label'))
        else:
            metric = xr.concat(all_metrics, pd.Index(args.label, name='label'))
    else:
        metric = xr.open_dataset(args.metric[0])
    metric = metric.to_dataframe().reset_index()

    last_axes = ''
    last_axes_label = ''
    last_plot = ''
    last_query = ''
    for i, (plot, axes, output_prefix) in enumerate(zip(args.plot, args.axes, args.output_prefix)):
        plot, last_plot = check_use_last_status(plot, last_plot)
        axes, last_axes = check_use_last_status(axes, last_axes)
        axes_label = None

        splitted_axes = axes.split('-')
        _metric = metric
        if i < len(args.query) and args.query[i]:
            query, last_query = check_use_last_status(args.query[i], last_query)
            _metric = _metric.query(query)

        plot_table[plot](_metric, *splitted_axes)

        if args.xlim:  plt.xlim(args.xlim)
        if args.ylim:  plt.ylim(args.ylim)
        if args.x_log: plt.xscale('log')
        if args.y_log: plt.yscale('log')
        if args.axes_label and i < len(args.axes_label):
            axes_label, last_axes_label = check_use_last_status(args.axes_label[i], last_axes_label)
        if axes_label is not None or last_axes_label != '':
            y_label, x_label = axes_label.split('-')
            plt.xlabel(x_label)
            plt.ylabel(y_label)

        for ext in args.format:
            plt.savefig(output_prefix + '.' + ext)
        plt.clf()

def check_use_last_status(value, last_status):
    if value == '*':
        assert last_status
        value = last_status
    else:
        last_status = value
    return value, last_status

@addplot('plot')
def plot(metric, y_axis, x_axis, c_axis=None):
    if c_axis:
        plot_multiple(metric, y_axis, x_axis, c_axis, np.unique(metric[c_axis]))
    else:
        y_max, y_max_x_value = plot_single(metric, y_axis, x_axis)
        if y_max < 1.0:
            plt.ylim([0.0, 1.0])
        plt.title('Max {}={:.3f}, {}={:.3f}'.format(y_axis, y_max, x_axis, y_max_x_value))
        
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

def plot_single(metric, y_axis, x_axis, label=None):
    xs = np.unique(metric[x_axis])
    ys = []
    for x in xs:
        query_format = '{}=="{}"' if isinstance(x, str) else '{}=={}'
        mean_y = np.mean(metric.query(query_format.format(x_axis, x))[y_axis])
        if np.isnan(mean_y): mean_y = 0
        ys.append(mean_y)
    plt.plot(xs, ys, label=label)
    y_max_index = np.argmax(ys)
    y_max = ys[y_max_index]
    y_max_x_value = xs[y_max_index]
    return y_max, y_max_x_value

def plot_multiple(metric, y_axis, x_axis, c_axis, cs):
    is_ylim_0to1 = False
    for c in cs:
        query_format = '{}=="{}"' if isinstance(c, str) else '{}=={}'
        query_metric = metric.query(query_format.format(c_axis, c))
        y_max, _ = plot_single(query_metric, y_axis, x_axis, label=c)
        is_ylim_0to1 = is_ylim_0to1 or y_max < 1.0
    
    if is_ylim_0to1:
        plt.ylim([0.0, 1.0])
    plt.legend()

@addplot('roc')
def plot_roc(metric, y_axis, x_axis, th_axis='threshold'):
    unique_thresholds = np.sort(np.unique(metric[th_axis]))

    xs = []
    ys = []
    for th in unique_thresholds:
        query_format = '{}=={}'
        _metric = metric.query(query_format.format(th_axis, th))
        mean_x = np.mean(_metric[x_axis])
        mean_y = np.mean(_metric[y_axis])
        xs.append(mean_x)
        ys.append(mean_y)
    
    plt.plot(xs, ys, marker='.')
    plt.text(xs[0], ys[0], '{:.3f}'.format(unique_thresholds[0]))
    plt.text(xs[-1], ys[-1], '{:.3f}'.format(unique_thresholds[-1]))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title('AUC:{:.3f}'.format(computeAUC(xs, ys)))

@addplot('multi-roc')
def plot_multi_roc(metric, y_axis, x_axis, c_axis, th_axis='threshold'):
    unique_thresholds = np.sort(np.unique(metric[th_axis]))
    unique_cs = np.unique(metric[c_axis])
    
    for c in unique_cs:
        query_format = '{}=="{}"' if isinstance(c, str) else '{}=={}'
        _metric = metric.query(query_format.format(c_axis, c))
        xs = []
        ys = []
        for th in unique_thresholds:
            query_format = '{}=={}'
            cur_metric = _metric.query(query_format.format(th_axis, th))
            mean_x = np.mean(cur_metric[x_axis])
            mean_y = np.mean(cur_metric[y_axis])
            xs.append(mean_x)
            ys.append(mean_y)

        plt.plot(xs, ys, marker='.', label=c)
        plt.text(xs[0], ys[0], '{:.3f}'.format(unique_thresholds[0]))
        plt.text(xs[-1], ys[-1], '{:.3f}'.format(unique_thresholds[-1]))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title('AUC:{:.3f}'.format(computeAUC(xs, ys)))
    plt.legend()
    
def computeAUC(xs, ys):
    area = 0.0
    for x0, y0, x1, y1 in zip(xs, ys, xs[1:], ys[1:]):
        area += abs((y0 + y1) * (x0 - x1) / 2)
    return area

@addplot('boxplot')
def boxplot(metric, y_axis, x_axis, c_axis=None):
    sns.boxplot(data=metric, x=x_axis, y=y_axis, hue=c_axis)
    if np.max(metric[y_axis]) < 1.0:
        plt.ylim([0.0, 1.0])

if __name__ == '__main__':
    main()
