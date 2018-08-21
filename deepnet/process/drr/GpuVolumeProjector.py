import numpy as np

import pydrr
import pydrr.utils as utils

import chainer.cuda

import pycuda.driver

import deepnet.utils
from deepnet.core import config

pycuda_device  = None
pycuda_context = None

class GpuVolumeProjector:
    def __init__(self,
        reverse_spacing = True,
        SOD = 1800,
        SDD = 2000,
        pixel_spacing = (0.32, 0.32),
        image_size = (512, 512),
        n_const_pose = None,
        gpu_id = 1,
        ):
        self.cache = {}
        self.image_size = image_size
        self.pixel_spacing = pixel_spacing
        self.reverse_spacing = reverse_spacing
        self.geometry_context = pydrr.GeometryContext()
        self.geometry_context.SOD = SOD
        self.geometry_context.SDD = SDD
        self.geometry_context.pixel_spacing = pixel_spacing
        self.geometry_context.image_size = image_size

        self.n_const_pose = n_const_pose

        if self.n_const_pose is not None:
            self.construct_detector(n_const_pose)

        global pycuda_context
        global pycuda_device
        if pycuda_context is None:
            pycuda_device = pycuda.driver.Device(gpu_id)
            pycuda_context = pycuda_device.make_context()
            pydrr.initialize()

    def __call__(
            self,
            myu_volume, spacing,
            case_name = None,
            pose=[],
        ):
        cupy_device = chainer.cuda.get_device_from_array(myu_volume)

        if self.reverse_spacing:
            spacing = spacing[::-1]

        volume = None
        if case_name is not None and case_name in self.cache:
            volume = self.cache[case_name]
            pycuda_context.push()
        else:
            myu_volume = deepnet.utils.unwrapped(myu_volume)

            pycuda_context.push()

            myu_volume = np.squeeze(myu_volume)
            if myu_volume.ndim == 4:
                myu_volume = myu_volume[0]
            volume = pydrr.VolumeContext(myu_volume, spacing).to_texture()
            if case_name is not None:
                self.cache[case_name] = volume

        if self.n_const_pose is None:
            self.construct_detector(len(pose))

        T_Nx4x4 = pydrr.utils.convertTransRotTo4x4(pose)

        d_image = self.projector.project(
            volume,
            self.geometry_context,
            T_Nx4x4
            )

        image = d_image.get()
        pycuda_context.pop()

        cupy_device.synchronize()

        return image

    def construct_detector(self, n_pose):
        n_channels = n_pose
        self.detector = pydrr.Detector(
            pydrr.Detector.make_detector_size(self.image_size, n_channels),
            self.pixel_spacing
            )
        self.projector = pydrr.Projector(self.detector, 1.0).to_gpu()
        return image