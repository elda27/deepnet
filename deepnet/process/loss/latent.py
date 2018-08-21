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


@register_process('loss.gradient_correlation')
def gradient_correlation(x, t, normalize = True, absolute = False):
    assert x.ndim == t.ndim
    xp = chainer.cuda.get_array_module(x)
    n_grad_dim = x.ndim - 2

    gc = []
    for i in range(n_grad_dim):
        kernel_shape = tuple( np.roll( (3,) + (1, ) * (n_grad_dim - 1), shift=i))
        w = xp.array([-1.0, 0.0, 1.0]).reshape((1, 1,) + kernel_shape)
        x_grad = F.convolution_nd(x, w)
        t_grad = F.convolution_nd(t, w)

        x_grad_mean = F.mean(x_grad, axis=tuple(range(1, x.ndim)))
        t_grad_mean = F.mean(t_grad, axis=tuple(range(1, x.ndim)))

        repeat_shape = (1, ) + x_grad.shape[1:]
        x_grad_mean = F.reshape(F.repeat(x_grad_mean, functools.reduce(lambda x, y: x * y, repeat_shape)), x_grad.shape)
        t_grad_mean = F.reshape(F.repeat(t_grad_mean, functools.reduce(lambda x, y: x * y, repeat_shape)), x_grad.shape)
        
        x_norm_grad = x_grad - x_grad_mean
        t_norm_grad = t_grad - t_grad_mean
        
        gc.append( F.sum( x_norm_grad * t_norm_grad ) / (F.sqrt(F.sum(x_norm_grad ** 2)) * F.sqrt(F.sum(t_norm_grad ** 2))) )

    return F.absolute(sum(gc) / len(gc))

@register_process("loss.penalty_sparse_encoding")
def penalty_sparse_encoding(vector, rho=0.05):
    h = F.mean(vector, axis=0)
    return F.sum(rho * F.log(rho / h) + (1 - rho) * F.log((1 - rho) / (1 - h)))
