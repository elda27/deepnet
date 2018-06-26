from abc import abstractmethod
import chainer
from chainer import functions as F
import numpy as np
try:
    import cupy as cp
    IMPORTED_CUPY=True
except ImportError:
    IMPORTED_CUPY=False
    pass
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import cv2
import warnings
import math
from functools import reduce
from itertools import cycle
import copy
import os
import os.path

from deepnet.utils import mhd
from deepnet import utils

RegistedVisualizers = {}

def register_visualizer(name):
    def _register_visualizer(func):
        RegistedVisualizers[name] = func
        return func
    return _register_visualizer

def create_visualizer(name):
    if name in RegistedVisualizers:
        raise ValueError('Unknown visualizer: {}'.format(name))
    return RegistedVisualizers[name]

def save_image(filename, image, spacing):
    _, ext = os.path.splitext(filename)
    if ext in ('.png', '.jpg', '.jpeg'):
        imageio.imwrite(filename, image)
    elif ext in ('.mhd', '.mha'):
        mhd.write(filename, image, { 'ElementSpacing': spacing })

class Visualizer:
    def __init__(self, output_filename):
        self.output_filename = output_filename
        self.is_last_finished = False

    def __call__(self, variables):
        self.variables = variables
        is_finished = self.render(variables)
        if is_finished and not self.is_last_finished:
            self.save()
        self.is_last_finished = is_finished

    def save(self):
        fig = self.get_figure()
        if fig is not None:
            imageio.imwrite(self.output_filename.format(**self.variables), fig)

    def to_np(self, image):
        if isinstance(image, chainer.Variable):
            image = np.array(chainer.functions.copy(image, -1).data)
        elif IMPORTED_CUPY and isinstance(image, cp.ndarray):
            image = cp.asnumpy(image)
        else:
            image =  np.array(image)
        return image

    @abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abstractmethod
    def render(self, variables):
        raise NotImplementedError()

    @abstractmethod
    def get_figure(self):
        '''
        @return image of visualize result.
        '''
        raise NotImplementedError()

@register_visualizer('write')
class ImageWriter(Visualizer):
    def __init__(self, output_filename, num_images, image_names):
        self.num_images = num_images
        self.image_names = image_names
        super().__init__(output_filename)

    def render(self, variables):
        iteration = variables['__iteration__']
        if iteration >= self.num_images:
            return True
        elif iteration == 0:
            self.images = dict()
        
        for image_name in self.image_names:
            assert image_name in variables
            image = self.to_np(variables[image_name])
            for i in range(image.shape[0]):
                if image_name in self.images and len(self.images[image_name]) >= self.num_images:
                    break
                self.images.setdefault(image_name, []).append(np.copy(image[i]))
        return False

    def save(self):
        figs = self.get_figure()
        for name, images in figs.items():
            for i, image in enumerate(images):
                spacing = None
                if 'spacing' in self.variables:
                    spacing = self.variables['spacing'][i]
                    if len(spacing) < img.ndim:
                        spacing = tuple(spacing) + (1,) * (img.ndim - len(spacing))

                save_image(self.output_filename.format(**self.variables, __index__=i, __name__=name), image, spacing)

    def get_figure(self):
        return self.images

@register_visualizer('mhd_write')
class MhdImageWriter(Visualizer):
    def __init__(self, output_filename, num_images, image_names):
        self.num_images = num_images
        self.image_names = image_names
        super().__init__(output_filename)

    def render(self, variables):
        iteration = variables['__iteration__']
        if iteration >= self.num_images:
            return True
        elif iteration == 0:
            self.images = dict()
        
        for image_name in self.image_names:
            assert image_name in variables
            image = self.to_np(variables[image_name])
            for i in range(image.shape[0]):
                if image_name in self.images and len(self.images[image_name]) >= self.num_images:
                    break
                self.images.setdefault(image_name, []).append(np.copy(image[i]))
        return False

    def save(self):
        figs = self.get_figure()
        for name, images in figs.items():
            for i, image in enumerate(images):
                header = {}
                img = np.squeeze(image)
                if 'spacing' in self.variables:
                    spacing = self.variables['spacing'][i]
                    if len(spacing) < img.ndim:
                        spacing = tuple(spacing) + (1,) * (img.ndim - len(spacing))
                    header = dict(
                        ElementSpacing=spacing
                    )
                mhd.write(self.output_filename.format(**self.variables, __index__=i, __name__=name), img, header)

    def get_figure(self):
        return self.images

class TileImageVisualizer(Visualizer):
    def __init__(self, output_filename, tile_shape, block_images, block_shape):
        self.num_block = tile_shape[0] * tile_shape[1]
        self.tile_shape = tile_shape
        self.block_images = block_images
        self.block_shape = block_shape
        super().__init__(output_filename)
    
    def render(self, variables):
        '''
            The image shape must be (N, C, H, W).
            This function accept batch images.
        '''
        iteration = variables['__iteration__']
        if iteration >= self.num_block:
            return True
        elif iteration == 0:
            self.images = []

        block_images_list = None
        batch_block_images = []
        for image_name in self.block_images:
            batch_block_images.append(self.to_np(variables[image_name]))
            if block_images_list is None:
                block_images_list = []
                for i in range(batch_block_images[-1].shape[0]):
                    block_images_list.append([])

        for block in batch_block_images:
            block = np.transpose(block, (0, 3, 2, 1))
            for j in range(block.shape[0]):
                block_images_list[j].append((self.normalize(block[j]) * 255).astype(np.uint8))

        for block_images in block_images_list:
            for i in range(0, reduce(lambda x, y: x * y, self.block_shape) - len(self.block_images)):
                block_images.append(np.zeros_like(self.images[-1]))

            self.images.append(self.make_tile_2d(block_images, self.block_shape))
        return False

    def get_figure(self):
        return self.make_tile_2d(self.images, self.tile_shape)

    def make_tile_2d(self, images, shape):
        assert len(shape) == 2, 'The shape of tile image has two dimension. {}, Length{}'.format(shape, len(shape))
        assert not (shape[0] is None and shape[1] is None), 'The element of shape is either one must not be None'
    
        num_images = len(images)
        tile_shape = shape
        if shape[0] is None:
            tile_shape = (math.ceil(num_images / shape[1]), shape[1])
        elif shape[1] is None:
            tile_shape = (shape[0], math.ceil(num_images / shape[0]))
        
        col_shapes = [0] * tile_shape[0]
        row_shapes = [0] * tile_shape[1]
        for y in range(tile_shape[1]):
            for x in range(tile_shape[0]):
                index = y + x * tile_shape[1]
                if len(images) <= index:
                    continue
                col_shapes[x] = max(images[index].shape[0], col_shapes[x])
                row_shapes[y] = max(images[index].shape[1], row_shapes[y])
        
        output_image = np.zeros([sum(col_shapes), sum(row_shapes), 3])
        
        for h in range(tile_shape[1]):
            for w in range(tile_shape[0]):
                index = h + tile_shape[1] * w
                if len(images) <= index:
                    continue
                image = images[index]
                if image.ndim == 2:
                    image = np.tile(image[:, :, np.newaxis], (1,1,3))
                left = sum(col_shapes[:w])
                top = sum(row_shapes[:h])
                output_image[left: left + image.shape[0],top: top + image.shape[1],:] = image
        return output_image
    
    def normalize(self, img):
        max_pix = np.amax(img)
        min_pix = np.amin(img)
        return (img.astype(np.float32) - min_pix) / (max_pix - min_pix + 1e-8)

@register_visualizer('nch_visualizer')
class NchImageVisualizer(TileImageVisualizer):
    def __init__(self, output_filename, num_rows, n_ch, n_ch_images, overlap_images, color_pallete, threshold=0.1, subtract=None):
        self.num_rows = num_rows
        self.n_ch = n_ch
        self.n_ch_images = n_ch_images if isinstance(n_ch_images, list) else [ n_ch_images ]
        self.overlap_images = overlap_images if isinstance(overlap_images, list) else [ overlap_images ]
        self.color_pallete = color_pallete
        self.subtract_images = subtract

        self.representation_vars = []
        for n_ch_image, overlap_image in zip(n_ch_images, overlap_images):
            self.representation_vars.append(overlap_image)
            for i in range(n_ch):
                self.representation_vars.append('NchImageVisualizer.{}.__{}_ch_images'.format(n_ch_image, i))

        # For subtraction images
        if subtract is not None:
            self.subtract_images = subtract
        else:
            self.subtract_images = []

        for subtract_pair in self.subtract_images:
            assert len(subtract_pair) == 2, 'Subtract images must be paired.{}'.format(subtract_pair)
            pair_string = '-'.join(subtract_pair)
            self.representation_vars.append('NchImageVisualizer.legend.' + pair_string)
            for i in range(n_ch):
                self.representation_vars.append('NchImageVisualizer.{}.__{}_ch_subtract_images'.format(pair_string, i))


        super().__init__(output_filename, (num_rows, 1), self.representation_vars, (len(self.overlap_images) + len(self.subtract_images), 1 + n_ch))
    
    def render(self, variables):
        #variables = copy.deepcopy(variables)
        iteration = variables['__iteration__']
        if iteration >= self.num_block:
            return True

        processed_image = {}

        for n_ch_image in self.n_ch_images:
            images = variables[n_ch_image]
            if isinstance(images, list):
                images = chainer.functions.concat([ F.expand_dims(img, axis=0) for img in images ], axis=0)
            
            images = F.transpose(images, (0, 1, 3, 2))
            processed_image[n_ch_image] = images
            for i, color in zip(range(self.n_ch), cycle(self.color_pallete)):
                index = 'NchImageVisualizer.{}.__{}_ch_images'.format(n_ch_image, i)
                variables[index] = chainer.functions.concat((
                    F.expand_dims(images[:,i] * color[0], axis=1), 
                    F.expand_dims(images[:,i] * color[1], axis=1), 
                    F.expand_dims(images[:,i] * color[2], axis=1),
                    ), axis=1)

        for subtract_pair in self.subtract_images:
            assert subtract_pair[0] in processed_image
            assert subtract_pair[1] in processed_image
            pair_string = '-'.join(subtract_pair)
            # make subtract images
            lhs_img = utils.unwrapped(processed_image[subtract_pair[0]])
            rhs_img = utils.unwrapped(processed_image[subtract_pair[1]])
            subtract_img = lhs_img - rhs_img * 2
            abs_img = np.absolute(subtract_img)
            for i, color in zip(range(self.n_ch), cycle(self.color_pallete)):
                index = 'NchImageVisualizer.{}.__{}_ch_subtract_images'.format(pair_string, i)
                variables[index] = np.concatenate((
                    np.expand_dims(abs_img[:, i] * (subtract_img[:,i] > 0) * 255, axis=1), 
                    np.expand_dims(abs_img[:, i] * 0.0, axis=1), 
                    np.expand_dims(abs_img[:, i] * (subtract_img[:,i] < 0) * 255, axis=1),
                    ), axis=1).astype(np.uint8)

            # make legend image
            legend_image = np.zeros(subtract_img.shape[2:] + (3,))
            cv2.putText(legend_image, subtract_pair[0], (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))
            cv2.putText(legend_image, subtract_pair[1], (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
            variables['NchImageVisualizer.legend.' + pair_string] = np.repeat(np.expand_dims(np.transpose(legend_image, (2, 1, 0)), axis=0), subtract_img.shape[0], axis=0)
        return super().render(variables)


class NetworkArchitectureVisualizer(Visualizer):
    def __init__(self, output_filename, loss_names):
        self.loss_names = loss_names if isinstance(loss_names, list) else [ loss_names ]
        super().__init__(output_filename)

    def render(self, variables):
        if variables['__iteration__'] == 0 and variables['__train_iteration__'] == 0:
            from chainer import computational_graph as cg
            import subprocess

            for loss_name in self.loss_names:
                graph_filename = self.output_filename.format(__name__=loss_name, **self.variables)
                loss = self.variables[loss_name]
                with open(graph_filename, 'w+') as o:
                    o.write(cg.build_computational_graph((loss, )).dump())

                try:
                    subprocess.call('dot -T png {} -o {}'.format(graph_filename, 
                                    graph_filename.replace('.dot', '.png')), 
                                    shell=True)
                except:
                    warnings.warn('please install graphviz and set your environment.')
            

        return True

    def get_figure(self):
        return None

class PlotVisualizerBase(Visualizer):
    dataframe = None
    #figure = plt.figure()
    def __init__(self, x, y, c = None, **kwargs):
        self.x = x
        self.y = y
        self.c = c
        self.image = None
        if isinstance(y, list):
            self.plot_method = self.plot_multiple_y
        else:
            self.plot_method = self.plot
    
    def __call__(self):
        plt.clf()
        self.plot_method(self.x, self.y, self.c, PlotVisualizer.dataframe)
        self.image = self.get_render_result()
    
    def get_render_result(self):
        fig = PlotVisualizer.figure
        fig.canvas.draw()
    
        w,h = fig.canvas.get_width_height()
        buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
        buf.shape = (w, h, 4)
    
        buf = np.roll ( buf, 3, axis = 2 )

        return buf

    def get_figure(self):
        return self.figure

    @abstractmethod
    def plot(self, x, y, c, df):
        raise NotImplementedError()

    @abstractmethod
    def plot_multiple_y(self, x, y, c, df):
        raise NotImplementedError()

class PlotVisualizer(PlotVisualizerBase):
    def plot(self, x, y, c, df):
        plt.plot(df[x], df[y])

    def plot_multiple_y(self, x, y, c, df):
        for y_ in y:
            self.plot(x, y_, None, df)
