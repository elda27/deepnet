import numpy as np
import chainer
from chainer.backends import cuda
from . import utils


def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    xp = cuda.get_array_module(x)
    theta = xp.pi / 180 * xp.random.uniform(-rg, rg)
    rotation_matrix = xp.array([[xp.cos(theta), -xp.sin(theta), 0],
                                [xp.sin(theta), xp.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.
    # Arguments
        x: input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Shifted Numpy image tensor.
    """
    xp = cuda.get_array_module(x)
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = xp.random.uniform(-hrg, hrg) * h
    ty = xp.random.uniform(-wrg, wrg) * w
    translation_matrix = xp.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shear(x, intensity, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shear of a Numpy image tensor.
    # Arguments
        x: input tensor. Must be 3D.
        intensity: Transformation intensity.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Sheared Numpy image tensor.
    """
    xp = cuda.get_array_module(x)
    shear = xp.random.uniform(-intensity, intensity)
    shear_matrix = xp.array([[1, -xp.sin(shear), 0],
                             [0, xp.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0.):
    """Performs a random spatial zoom of a Numpy image tensor.
    # Arguments
        x: input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Zoomed Numpy image tensor.
    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    xp = cuda.get_array_module(x)
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = xp.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = xp.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def transform_matrix_offset_center(matrix, x, y):
    xp = cuda.get_array_module(x)

    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = xp.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = xp.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = xp.dot(xp.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.,
                    interp_order=0):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    xp = cuda.get_array_module(x)
    x = xp.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [utils.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = xp.stack(channel_images, axis=0)
    x = xp.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x, axis):
    xp = cuda.get_array_module(x)
    x = xp.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


class ImageDataGenerator(object):
    """Generate image data with data augmentation.
    Arguments:
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        intensity_range: intensity (0 to Inf).
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
    """

    def __init__(self,
                 rotation_range=0.0,
                 width_shift_range=0.0,
                 height_shift_range=0.0,
                 shear_range=0.0,
                 zoom_range=0.0,
                 intensity_range=0.0,
                 cval=0.0,
                 horizontal_flip=False,
                 vertical_flip=False,
                 keep_aspect=True
                 ):

        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.intensity_range = intensity_range
        self.cval = cval

        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        self.row_axis = 1
        self.col_axis = 2
        self.channel_axis = 3

        self.interp_order = 1
        self.fill_mode = 'nearest'
        self.keep_aspect = keep_aspect

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def random_transform(self, x):
        xp = cuda.get_array_module(x)

        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        # if seed is not None:
        #     xp.random.seed(seed)

        input_shape = x.shape

        # """ set random transform value"""
        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            self.theta = xp.pi / 180 * \
                xp.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            self.theta = 0

        if self.height_shift_range:
            self.tx = xp.random.uniform(-self.height_shift_range,
                                        self.height_shift_range) * input_shape[img_row_axis]
        else:
            self.tx = 0

        if self.width_shift_range:
            self.ty = xp.random.uniform(-self.width_shift_range,
                                        self.width_shift_range) * input_shape[img_col_axis]
        else:
            self.ty = 0

        if self.shear_range:
            self.shear = xp.random.uniform(-self.shear_range, self.shear_range)
        else:
            self.shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            self.zx, self.zy = 1, 1
        else:
            self.zx, self.zy = xp.random.uniform(
                self.zoom_range[0], self.zoom_range[1], 2)
        if self.keep_aspect:
            self.zy = self.zx

        if self.horizontal_flip:
            self.horizontal_flip_prob = xp.random.random()
        else:
            self.horizontal_flip_prob = 0

        if self.vertical_flip:
            self.vertical_flip_prob = xp.random.random()
        else:
            self.vertical_flip_prob = 0

        self.transform_matrix = None
        if self.theta != 0:
            rotation_matrix = xp.array([[xp.cos(self.theta), -xp.sin(self.theta), 0],
                                        [xp.sin(self.theta), xp.cos(
                                            self.theta), 0],
                                        [0, 0, 1]], dtype=xp.float32)
            self.transform_matrix = rotation_matrix

        if self.tx != 0 or self.ty != 0:
            shift_matrix = xp.array([[1, 0, self.tx],
                                     [0, 1, self.ty],
                                     [0, 0, 1]], dtype=xp.float32)
            self.transform_matrix = shift_matrix if self.transform_matrix is None else xp.dot(
                self.transform_matrix, shift_matrix)

        if self.shear != 0:
            shear_matrix = xp.array([[1, -xp.sin(self.shear), 0],
                                     [0, xp.cos(self.shear), 0],
                                     [0, 0, 1]], dtype=xp.float32)
            self.transform_matrix = shear_matrix if self.transform_matrix is None else xp.dot(
                self.transform_matrix, shear_matrix)

        if self.zx != 1 or self.zy != 1:
            zoom_matrix = xp.array([[self.zx, 0, 0],
                                    [0, self.zy, 0],
                                    [0, 0, 1]], dtype=xp.float32)
            self.transform_matrix = zoom_matrix if self.transform_matrix is None else xp.dot(
                self.transform_matrix, zoom_matrix)

        if self.transform_matrix is not None:
            h, w = input_shape[img_row_axis], input_shape[img_col_axis]
            self.transform_matrix = transform_matrix_offset_center(
                self.transform_matrix, h, w)

        if self.intensity_range != 0:
            self.intensity = xp.random.uniform(
                -self.intensity_range, self.intensity_range)
        else:
            self.intensity = 0.

        # """ apply transform """
        return self.fixed_transform(x)

    def fixed_transform(self, input, label=False):
        old_interp_order = self.interp_order
        old_constant = self.cval

        try:
            if label:
                self.interp_order = 0
                self.c_val = 0
            return self.fixed_transform_(input)
        finally:
            self.interp_order = old_interp_order
            self.cval = old_constant

    def fixed_transform_(self, input):
        x = None
        xp = cuda.get_array_module(input)
        if isinstance(x, chainer.Variable):
            x = xp.copy(input.data.astype(xp.float32))
        else:
            x = xp.copy(input.astype(xp.float32))

        xp = cuda.get_array_module(x)

        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if self.transform_matrix is not None:
            x = apply_transform(x, self.transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode,
                                cval=self.cval,
                                interp_order=self.interp_order)

        if self.horizontal_flip_prob > 0.5:
            x = flip_axis(x, img_col_axis)

        if self.vertical_flip_prob > 0.5:
            x = flip_axis(x, img_row_axis)

        x = x + self.intensity

        return chainer.Variable(x)
