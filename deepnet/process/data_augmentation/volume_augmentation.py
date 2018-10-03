import numpy as np
import chainer.cuda
from . import utils


def flip_axis(x, axis):
    xp = chainer.cuda.get_array_module(x)
    x = xp.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def transform_matrix_offset_center(matrix, x, y, z):
    xp = chainer.cuda.get_array_module(matrix)
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    o_z = float(z) / 2 + 0.5
    offset_matrix = xp.array([[1, 0, 0, o_x], [0, 1, 0, o_y], [
                             0, 0, 1, o_z], [0, 0, 0, 1]], dtype=xp.float32)
    reset_matrix = xp.array(
        [[1, 0, 0, -o_x], [0, 1, 0, -o_y], [0, 0, 1, -o_z], [0, 0, 0, 1]], dtype=xp.float32)
    transform_matrix = xp.dot(xp.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, fill_mode='nearest', cval=0., interp_order=0):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 3D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interp_order: the order of the spline interpolation
    # Returns
        The transformed version of the input.
    """
    xp = chainer.cuda.get_array_module(x)

    final_affine_matrix = transform_matrix[:3, :3]
    final_offset = transform_matrix[:3, -1]

    transformed_volume = utils.affine_transform(
        x,
        final_affine_matrix,
        final_offset,
        order=interp_order,  # The order of the spline interpolation
        mode=fill_mode,
        cval=cval)

    return transformed_volume


def apply_crop(img, center, size):

    crop_x = size[0]
    crop_y = size[1]
    crop_z = size[2]

    start_x = center[0]-crop_x//2
    start_y = center[1]-crop_y//2
    start_z = center[2]-crop_z//2

    return img[start_x:start_x+crop_x, start_y:start_y+crop_y, start_z:start_z+crop_z]


class VolumeDataGenerator(object):
    """Generate volumetric data with data augmentation.
    # Arguments
        rotation_range: degrees (0 to 180).
        translation_range: fraction of  volume size.
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        intensity_range: intensity (0 to Inf).
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        interp_order: Spline order. If the order is zero, the interpolation will be
                      nearest neigbor. Default is 1.
        random_crop: crop input volume randomly.
        crop_size: crop to specific size.
        crop_order: the order to crop. If set to 'pre', geometric transformation
            will be performed after cropping. This is effective for reducing computation time.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        depth_flip: whether to randomly flip images in the depth.
    """

    def __init__(self,
                 rotation_range=0.,    # [deg]
                 translation_range=0.,  # [%]
                 zoom_range=0.,        # [%]
                 intensity_range=0.,   # [intensity]
                 fill_mode='nearest',
                 horizontal_flip=False,
                 vertical_flip=False,
                 depth_flip=False,
                 cval=0.,
                 interp_order=1,
                 random_crop=False,
                 crop_size=[100, 100, 100],
                 crop_order='pre',
                 ):

        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.zoom_range = zoom_range
        self.intensity_range = intensity_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.interp_order = interp_order
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.depth_flip = depth_flip
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.crop_order = crop_order

        self.row_axis = 1
        self.col_axis = 2
        self.z_axis = 3

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def reset_center(self, x):
        input_shape = x.shape
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_z_axis = self.z_axis - 1
        if self.crop_order == 'pre' and self.random_crop:
            h, w, z = self.crop_size
        else:
            h, w, z = input_shape[img_row_axis], input_shape[img_col_axis], input_shape[img_z_axis]
        self.transform_matrix = transform_matrix_offset_center(
            self.transform_matrix_origin.copy(), h, w, z)

    def random_transform(self, x):
        """Randomly augment a single image tensor.
        # Arguments
            x: 4D tensor, single image.
        # Returns
            A randomly transformed version of the input (same shape or cropped shape).
        """

        xp = chainer.cuda.get_array_module(x)

        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_z_axis = self.z_axis - 1

        if x is not None:
            input_shape = x.shape

        if self.crop_order == 'pre' and self.random_crop:
            h, w, z = self.crop_size
        else:
            h, w, z = input_shape[img_row_axis], input_shape[img_col_axis], input_shape[img_z_axis]

        if self.rotation_range:
            px = xp.pi / 180 * \
                xp.random.uniform(-self.rotation_range, self.rotation_range)
            py = xp.pi / 180 * \
                xp.random.uniform(-self.rotation_range, self.rotation_range)
            pz = xp.pi / 180 * \
                xp.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            px = py = pz = 0

        if self.translation_range:
            tx = xp.random.uniform(-self.translation_range,
                                   self.translation_range) * h
            ty = xp.random.uniform(-self.translation_range,
                                   self.translation_range) * w
            tz = xp.random.uniform(-self.translation_range,
                                   self.translation_range) * z
        else:
            tx = ty = tz = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy, zz = 1, 1, 1
        else:
            zx, zy, zz = xp.random.uniform(
                self.zoom_range[0], self.zoom_range[1], 3)

        self.transform_matrix = xp.eye(4, 4, dtype=xp.float32)

        if px != 0 or py != 0 or pz != 0:
            Rx = xp.array([[1, 0, 0],
                           [0, xp.cos(px), xp.sin(px)],
                           [0, -xp.sin(px), xp.cos(px)]], dtype=xp.float32)
            Ry = xp.array([[xp.cos(py), 0, -xp.sin(py)],
                           [0, 1, 0],
                           [xp.sin(py), 0, xp.cos(py)]], dtype=xp.float32)
            Rz = xp.array([[xp.cos(pz), xp.sin(pz), 0],
                           [-xp.sin(pz), xp.cos(pz), 0],
                           [0, 0, 1]], dtype=xp.float32)

            rotation_matrix = xp.zeros((4, 4), dtype=xp.float32)
            rotation_matrix[:3, :3] = Rz.dot(Ry).dot(Rx)  # z-y-x
            rotation_matrix[-1, -1] = 1

            self.transform_matrix = rotation_matrix

        if tx != 0 or ty != 0 or tz != 0:
            Txyz = xp.array([[1, 0, 0, tx],
                             [0, 1, 0, ty],
                             [0, 0, 1, tz],
                             [0, 0, 0, 1]], dtype=xp.float32)

            self.transform_matrix = Txyz if self.transform_matrix is None else xp.dot(
                self.transform_matrix, Txyz)

        if zx != 1 or zy != 1 or zz != 1:
            zoom_matrix = xp.array([[zx, 0, 0, 0],
                                    [0, zy, 0, 0],
                                    [0, 0, zz, 0],
                                    [0, 0, 0, 1]], dtype=xp.float32)

            self.transform_matrix = zoom_matrix if self.transform_matrix is None else xp.dot(
                self.transform_matrix, zoom_matrix)

        self.transform_matrix_origin = self.transform_matrix.copy()
        if self.transform_matrix is not None:
            self.transform_matrix = transform_matrix_offset_center(
                self.transform_matrix, h, w, z)

        if self.horizontal_flip:
            self.horizontal_flip_prob = xp.random.random()
        else:
            self.horizontal_flip_prob = 0

        if self.vertical_flip:
            self.vertical_flip_prob = xp.random.random()
        else:
            self.vertical_flip_prob = 0

        if self.depth_flip:
            self.depth_flip_prob = xp.random.random()
        else:
            self.depth_flip_prob = 0

        if self.random_crop:
            self.crop_center_x = xp.random.randint(
                self.crop_size[0]//2, input_shape[0]-self.crop_size[0]//2)
            self.crop_center_y = xp.random.randint(
                self.crop_size[1]//2, input_shape[1]-self.crop_size[1]//2)
            self.crop_center_z = xp.random.randint(
                self.crop_size[2]//2, input_shape[2]-self.crop_size[2]//2)

        if self.intensity_range != 0:
            self.intensity = xp.random.uniform(
                -self.intensity_range, self.intensity_range)
        else:
            self.intensity = 0.

        # 　""" apply transform """
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
        xp = chainer.cuda.get_array_module(input)
        x = None
        if isinstance(x, chainer.Variable):
            x = xp.copy(input.data)
        else:
            x = xp.copy(input)

        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_z_axis = self.z_axis - 1

        if self.crop_order == 'pre' and self.random_crop:
            x = apply_crop(
                x, [self.crop_center_x, self.crop_center_y, self.crop_center_z], self.crop_size)

        if self.transform_matrix is not None:
            x = apply_transform(x, self.transform_matrix,
                                fill_mode=self.fill_mode,
                                cval=self.cval,
                                interp_order=self.interp_order)

        if self.horizontal_flip_prob > 0.5:
            x = flip_axis(x, img_col_axis)

        if self.vertical_flip_prob > 0.5:
            x = flip_axis(x, img_row_axis)

        if self.depth_flip_prob > 0.5:
            x = flip_axis(x, img_z_axis)

        if self.crop_order == 'post' and self.random_crop:
            x = apply_crop(
                x, [self.crop_center_x, self.crop_center_y, self.crop_center_z], self.crop_size)

        x = x + self.intensity

        return chainer.Variable(x) if isinstance(input, chainer.Variable) else x
