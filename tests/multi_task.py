#!/usr/bin/env python

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import numpy as np
import cupy as xp
from chainer.backends import cuda

import argparse

from deepnet.process.loss.distance import l2_norm_distance
from deepnet.process.loss.multi_task import MultiTaskLoss

import os
import shutil
import warnings

from chainer import reporter
from chainer import serializer as serializer_module
from chainer.training import extension
from chainer.training import trigger as trigger_module
from chainer import utils

# network


class Net(chainer.Chain):
    def __init__(self, n_out=1, device=-1):
        super().__init__()

        initializer = chainer.initializers.Zero()

        self.iteration = 0
        with self.init_scope():
            self.l1 = L.Linear(None, n_out)
            self.l2 = L.Linear(None, n_out)
            self.multi_task_loss = MultiTaskLoss(
                ['euclidean', 'euclidean'], initialize=[10.0, 0.0])

    def __call__(self, x, t1, t2):

        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(x))

        # euclidean_distance(h1, t1, reduce=False, sqrt=lambda x: x)
        #d1 = F.sum((h1 - t1) ** 2)
        # euclidean_distance(h2, t2, reduce=False, sqrt=lambda x: x)
        #d2 = F.sum((h2 - t2) ** 2)
        d1 = l2_norm_distance(h1, t1, reduce='no')
        d2 = l2_norm_distance(h2, t2, reduce='no')
        loss = self.multi_task_loss(d1, d2)
        loss = F.mean(loss)

        chainer.reporter.report(
            {
                'loss': loss,
                'b1': float(F.exp(self.multi_task_loss.sigma_0).data ** 0.5),
                'b2': float(F.exp(self.multi_task_loss.sigma_1).data ** 0.5),
                'Lb1': float(self.l1.b.data),
                'Lb2': float(self.l2.b.data),
            },
            self
        )

        if self.iteration % 100 == 0:
            print('loss: {}, rs1:{}, s1: {}, rs2:{}, s2:{}'.format(
                loss.data,
                float(self.multi_task_loss.sigma_0.data),
                float(F.exp(self.multi_task_loss.sigma_0).data ** 0.5),
                float(self.multi_task_loss.sigma_1.data),
                float(F.exp(self.multi_task_loss.sigma_1).data ** 0.5),
            ))
        self.iteration += 1

        return loss


class MultiTasker(chainer.Chain):

    def __init__(self, n_out=1, device=-1):
        super(MultiTasker, self).__init__()

        initializer = chainer.initializers.Zero()

        self.iteration = 0
        with self.init_scope():

            self.l1 = L.Linear(None, n_out)
            self.l2 = L.Linear(None, n_out)

            self.b1 = L.Linear(None, 1, initialW=initializer, nobias=True)
            self.b2 = L.Linear(None, 1, initialW=initializer, nobias=True)

        self.one1 = cuda.to_gpu(np.ones((1, 1), np.float32), device=device)
        self.one2 = cuda.to_gpu(np.ones((1, 1), np.float32), device=device)

    def __call__(self, x, t1, t2):

        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(x))

        b1 = self.b1(self.one1)
        b2 = self.b2(self.one2)

        b1 = F.expand_dims(F.repeat(b1, h1.shape[0]), axis=1)
        b2 = F.expand_dims(F.repeat(b2, h2.shape[0]), axis=1)

        loss = F.sum(F.exp(-b1) * (h1 - t1)**2. + b1) + \
            F.sum(F.exp(-b2) * (h2 - t2)**2. + b2)
        loss = F.mean(loss)

        if self.iteration % 100 == 0:
            print('loss:', loss.data, ', sigma1:', F.exp(
                b1[0]).data[0]**0.5, ', sigma2:', F.exp(b2[0]).data[0]**0.5)

        chainer.reporter.report(
            {
                'loss': loss,
                'b1': float(F.exp(b1[0]).data ** 0.5),
                'b2': float(F.exp(b2[0]).data ** 0.5),
                'Lb1': float(self.l1.b.data),
                'Lb2': float(self.l2.b.data),
            },
            self
        )

        self.iteration += 1
        return loss


# dataset
class SyntheticDataset(chainer.dataset.DatasetMixin):

    def __init__(self, n_samples=2000, n_dim=1):

        X = np.random.randn(n_samples, n_dim)
        w1 = 2.
        b1 = 8.
        sigma1 = 1e3
        Y1 = X.dot(w1) + b1 + sigma1 * np.random.randn(n_samples, n_dim)
        w2 = 3
        b2 = 3.
        sigma2 = 1e0
        Y2 = X.dot(w2) + b2 + sigma2 * np.random.randn(n_samples, n_dim)

        self.X = X.astype(np.float32)
        self.Y1 = Y1.astype(np.float32)
        self.Y2 = Y2.astype(np.float32)

        self._n_samples = n_samples
        self._n_dim = n_dim

    def __len__(self):
        return len(self.X)

    def get_example(self, i):
        return self.X[i], self.Y1[i], self.Y2[i]


def print_log():
    pass


def main():
    parser = argparse.ArgumentParser(description='Chainer example: Multi-task')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=2000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    train = SyntheticDataset()
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test = SyntheticDataset()
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, shuffle=False, repeat=False)

    model = Net(device=args.gpu)
    #model = MultiTasker(device=args.gpu)

    chainer.backends.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)

    trainer.extend(CsvLogReport())
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
    trainer.extend(
        extensions.Evaluator(test_iter, model, device=args.gpu),
        trigger=(100, 'epoch'),
        name='valid'
    )
    # trainer.extend(
    #     extensions.PrintReport(
    #         ['epoch', 'main/loss', 'valid/main/loss', 'elapsed_time']
    #     ),
    #     trigger=(10, 'epoch')
    # )
    trainer.extend(extensions.ProgressBar())

    trainer.run()


class CsvLogReport(extension.Extension):
    def __init__(self, keys=None, trigger=(1, 'epoch'), postprocess=None,
                 log_name='log'):
        self._keys = keys
        self._trigger = trigger_module.get_trigger(trigger)
        self._postprocess = postprocess
        self._log_name = log_name
        self._log = []

        self._init_summary()

    def __call__(self, trainer):
        # accumulate the observations
        observation = trainer.observation
        summary = self._summary
        keys = self._keys

        if keys is None:
            summary.add(observation)
        else:
            summary.add({k: observation[k] for k in keys if k in observation})

        if self._trigger(trainer):
            stats = summary.compute_mean()

            updater = trainer.updater
            stats['epoch'] = updater.epoch
            stats['iteration'] = updater.iteration
            stats['elapsed_time'] = trainer.elapsed_time

            if self._keys is None:
                self._keys = list(stats.keys())

            new_path = os.path.join(trainer.out, self._log_name + '.csv')
            # reset the summary for the next output
            if os.path.exists(new_path):
                with open(new_path, 'a') as fp:
                    fp.write(','.join([str(float(stats[k]))
                                       for k in self._keys]) + '\n')
            else:
                with open(new_path, 'w+') as fp:
                    fp.write(','.join([k for k in self._keys]) + '\n')
                    fp.write(','.join([str(float(stats[k]))
                                       for k in self._keys]) + '\n')
            self._init_summary()

    def _init_summary(self):
        self._summary = reporter.DictSummary()


if __name__ == '__main__':
    main()
