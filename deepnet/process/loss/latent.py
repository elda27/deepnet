from deepnet.core.registration import register_process
import chainer
import chainer.functions as F
import numpy as np
import functools


@register_process('loss.constrain_kernel')
def constrain_kernel(network):
    n_kernels = 0
    norm = None
    for node in network.updatable_node:
        for lname, layer in node.model.layers:
            if isinstance(layer, (chainer.links.ConvolutionND, chainer.links.DeconvolutionND)):
                n_kernels += 1
                if norm is None:
                    norm = F.batch_l2_norm_squared(layer.W)
                else:
                    norm += F.batch_l2_norm_squared(layer.W)
    return norm / n_kernels


@register_process("loss.penalty_sparse_encoding")
def penalty_sparse_encoding(vector, rho=0.05):
    h = F.mean(vector, axis=0)
    return F.sum(rho * F.log(rho / h) + (1 - rho) * F.log((1 - rho) / (1 - h)))
