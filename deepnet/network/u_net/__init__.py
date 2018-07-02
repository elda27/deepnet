import chainer 
import chainer.functions as F
import chainer.links as L
from chainer import initializers
from chainer.utils import argument
import numpy as np
import warnings
from deepnet import utils
from deepnet.network.utils import bayesian_dropout, get_upsampling_filter_2d, get_upsampling_filter_3d
from deepnet.network.init import register_network

class UNetBlock(chainer.Chain):
    
    """ convolution blocks for use on U-Net """

    def __init__(self, 
                 n_dims, 
                 in_channels, 
                 hidden_channels, 
                 out_channel, 
                 kernel_size=3, 
                 initialW=initializers.HeNormal(), 
                 initial_bias=None, 
                 block_type='default', 
                 is_residual=False,
                 batch_norm=False):
        
        self.n_dims = n_dims
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channel = out_channel  
        self.kernel_size = kernel_size 
        self.initialW = initialW
        self.initial_bias = initial_bias
        self.block_type = block_type
        self.is_residual = is_residual
        self.batch_norm = batch_norm
        
        pad = self.kernel_size // 2 if self.kernel_size%2==0 else (self.kernel_size-1) // 2
        
        super().__init__()
        
        with self.init_scope():
        
            if self.block_type == 'default':
                self.conv_1=L.ConvolutionND(self.n_dims, self.in_channels, self.hidden_channels, self.kernel_size,  stride=1, pad=pad, initialW=self.initialW, initial_bias=self.initial_bias)
                self.conv_2=L.ConvolutionND(self.n_dims, self.hidden_channels, self.out_channel, self.kernel_size, stride=1, pad=pad, initialW=self.initialW, initial_bias=self.initial_bias)
            
            elif self.block_type == 'dilated':
                assert self.n_dims != 2, 'Currently, dilated convolution is unsupported in 3D.'
                self.conv_1=L.DilatedConvolution2D(self.in_channels, self.hidden_channels, self.kernel_size,  stride=1, pad=pad, dilate=1, initialW=self.initialW, initial_bias=self.initial_bias)
                self.conv_2=L.DilatedConvolution2D(self.hidden_channels, self.out_channel, self.kernel_size, stride=1, pad=pad*2, dilate=2, initialW=self.initialW, initial_bias=self.initial_bias)
            
            elif self.block_type == 'mlp':
                assert self.n_dims != 2, 'Currently, mlp convolution is unsupported in 3D.'
                self.conv_1=L.MLPConvolution2D(self.in_channels, [self.hidden_channels]*3, self.kernel_size,  stride=1, pad=pad, conv_init=self.initialW, bias_init=self.initial_bias)
                self.conv_2=L.MLPConvolution2D(self.hidden_channels, [self.out_channel]*3, self.kernel_size, stride=1, pad=pad, conv_init=self.initialW, bias_init=self.initial_bias)
    
            if self.batch_norm:               
                self.bn_conv_1=L.BatchNormalization(self.hidden_channels)
                self.bn_conv_2=L.BatchNormalization(self.out_channel)


    def __call__(self, x):
        
        if self.is_residual == True:
            
            h1 = self.conv_1(x)
            if self.batch_norm:
                h1 = self.bn_conv_1(h1)
            h1 = F.relu(h1)           
            h2 = self.conv_2(h1)
            if self.batch_norm:
                h2 = self.bn_conv_2(h2)
            h2 = F.relu(h2)
            
            return h1 + h2

        else:
            h = self.conv_1(x)
            if self.batch_norm:
                h = self.bn_conv_1(h)
            h = F.relu(h)  
            h = self.conv_2(h)
            if self.batch_norm:
                h = self.bn_conv_2(h)
            h = F.relu(h)
            
            return h

@register_network('network.U-net')
class UNet(chainer.Chain):
    """ Builds a U-Net architecture. """

    def __init__(self, 
                 in_channel,
                 out_channel,
                 n_dims=2, 
                 kernel_size=3, 
                 n_layers=5, 
                 n_filters=64, 
                 class_weight=[None], 
                 label_types=['categorical'],
                 is_bayesian=False, 
                 is_residual=False,
                 initialW=initializers.HeNormal(), 
                 initial_bias=None, 
                 batch_norm=False,
                 block_type='default',
                 **kwargs):

        self.n_dims = n_dims
        if self.n_dims != 2 and self.n_dims != 3:
            warnings.warn('unsupported number of input dimensions.')      
            
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.is_bayesian = is_bayesian
        self.is_residual = is_residual
        
        self.class_weight = class_weight
        if self.out_channel != len(class_weight) or not isinstance(class_weight, list):
            raise ValueError('out_channel != len(class_weight). Actual: {} != {}'.format(self.out_channel, len(class_weight))) 

        self.label_types = label_types
        if self.out_channel != len(label_types) or not isinstance(label_types, list):
            raise ValueError('out_channel != len(label_types). Actual: {} != {}'.format(self.out_channel, len(label_types)))     

        self.initialW = initialW
        self.initial_bias = initial_bias
        self.block_type = block_type
        self.batch_norm = batch_norm
        
        chainer.Chain.__init__(self)
                
        with self.init_scope():
            # down convolution
            for i in range(1, self.n_layers+1):
                if i == 1:
                    setattr(self, 'down_unet_block_%d' % i, UNetBlock(self.n_dims, self.in_channel, self.n_filters*(2**(i-1)), self.n_filters*(2**(i-1)), self.kernel_size, initialW=initialW, initial_bias=initial_bias, is_residual=self.is_residual, block_type=block_type, batch_norm=batch_norm))
                else:
                    setattr(self, 'down_unet_block_%d' % i, UNetBlock(self.n_dims, self.n_filters*(2**(i-2)), self.n_filters*(2**(i-1)), self.n_filters*(2**(i-1)), self.kernel_size, initialW=initialW, initial_bias=initial_bias, is_residual=self.is_residual, block_type=block_type, batch_norm=batch_norm))
        
            # up convolution
            for i in range(1, self.n_layers):
                deconv_n_filters = self['down_unet_block_%d' % (i+1)].out_channel
                setattr(self, 'deconv_%d' % i, L.DeconvolutionND(self.n_dims, deconv_n_filters, deconv_n_filters, self.kernel_size, stride=2, pad=0, initialW=initialW, initial_bias=initial_bias))
    
                if self.batch_norm:
                    setattr(self, 'bn_deconv_%d' % i, L.BatchNormalization(deconv_n_filters))
    
                upconv_n_filters = self['down_unet_block_%d' % i].out_channel + self['deconv_%d' % i].W.shape[1]               
                setattr(self, 'up_unet_block_%d' % i, UNetBlock(self.n_dims, upconv_n_filters, self.n_filters*(2**(i-1)), self.n_filters*(2**(i-1)), self.kernel_size, initialW=initialW, initial_bias=initial_bias, is_residual=self.is_residual, block_type=block_type, batch_norm=batch_norm))
                
                if i == 1: # output layer
                    setattr(self, 'up_conv%d_3' % i, L.ConvolutionND(self.n_dims, self.n_filters*(2**(i-1)), self.out_channel, ksize=self.kernel_size, stride=1, pad=1, initialW=initialW, initial_bias=initial_bias))    
    
        
            # initialize weights for deconv layer
            for i in range(1, self.n_layers):
                deconv_k_size    = self['deconv_%d' % i].W.shape[-1]
                deconv_n_filters = self['deconv_%d' % i].W.shape[1] 
    
                self['deconv_%d' % i].W.data[...] = 0
    
                if self.n_dims == 2:          
                    filt = get_upsampling_filter_2d(deconv_k_size)               
                    self['deconv_%d' % i].W.data[range(deconv_n_filters), range(deconv_n_filters), :, :] = filt     
                elif self.n_dims == 3:
                    filt = get_upsampling_filter_3d(deconv_k_size)
                    self['deconv_%d' % i].W.data[range(deconv_n_filters), range(deconv_n_filters), :, :, :] = filt                    
    

        #self.train = False


    def down_conv_activate_function(self, x):
        return F.leaky_relu(x, slope=0.0) # 0.01

    def up_conv_activate_function(self, x):
        return F.relu(x)

    def down_conv_dropout(self, x):
        if self.is_bayesian:
            return bayesian_dropout(x, ratio=0.5)
        else:
            return F.dropout(x, ratio=0.0)
        
    def up_conv_dropout(self, x):
        if self.is_bayesian:
            return bayesian_dropout(x, ratio=0.5)
        else:
            return F.dropout(x, ratio=0.0)

    def freeze_layers(self, startwith, verbose=False):
        for l in self.children():
            if l.name.startswith(startwith):
                l = getattr(self, l.name)
                l.disable_update()
                if verbose==True:
                    print(l.name, 'disable_update')  

    def __call__(self, x):
        
        store_activations = {}
        
        # down convolution
        for i in range(1, self.n_layers+1):
        
            if i == 1:
                h = F.identity(x)
            else:
                h = F.max_pooling_nd(h, 2, stride=2)
                
            h = self['down_unet_block_%d' % (i)](h)
            h = self.down_conv_dropout(h)
            store_activations['down_unet_block_%d' % (i)] = h
            
        del h # clear hidden layer
              
        # up convolution
        for i in range(self.n_layers-1, 0, -1):
            
            if i == self.n_layers-1:
                h = store_activations['down_unet_block_%d' % (i+1)]
                del store_activations['down_unet_block_%d' % (i+1)] # clear
            else:
                h = h 
                
            h = self['deconv_%d' % i](h)                
            if self.batch_norm:
                h = self['bn_deconv_%d' % i](h)                
            h = self.up_conv_activate_function(h)      
            down_conv = store_activations['down_unet_block_%d' % (i)]
            del store_activations['down_unet_block_%d' % (i)] # clear          

            if self.n_dims == 2:
                h = F.concat([h[:,:,0:down_conv.shape[2],0:down_conv.shape[3]], down_conv]) # fuse layer   
            elif self.n_dims == 3:
                h = F.concat([h[:,:,0:down_conv.shape[2],0:down_conv.shape[3],0:down_conv.shape[4]], down_conv]) # fuse layer   
            del down_conv

            h = self['up_unet_block_%d' % i](h)
            h = self.up_conv_dropout(h)
                
            if i == 1:
                o = self['up_conv%d_3' % i](h)
                if self.n_dims == 2:
                    score = o[:,:,0:x.shape[2],0:x.shape[3]] 
                elif self.n_dims == 3:
                    score = o[:,:,0:x.shape[2],0:x.shape[3],0:x.shape[4]] 

                self.score = score
                
        del h, o # clear hidden layer

        return self.score