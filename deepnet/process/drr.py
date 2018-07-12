import numpy as np
import pydrr
import pydrr.utils as utils

class VolumeProjector:
    def __init__(self, 
        reverse_spacing = True, 
        SOD = 1800,
        SDD = 2000,
        pixel_spacing = (0.32, 0.32),
        image_size = (1024, 1024),
        n_const_pose = None
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

    def __call__(self,
        myu_volume, spacing,
        case_name = None,
        pose=[],
        ):
        volume = None
        if case_name is not None and case_name in self.cache:
            volume = self.cache[case_name]
        else:
            volume = pydrr.VolumeContext(deepnet.utils.unwrapped(myu_volume), spacing).to_texture()
            if case_name is not None:
                self.cache[case_name] = volume

        if self.reverse_spacing:
            spacing = spacing[::-1]

        if self.n_const_pose is None:
            self.construct_detector(len(pose))

        T_Nx4x4 = pydrr.utils.convertTransRotTo4x4(pose)

        d_image = self.projector.project(
            volume,
            self.geometry_context,
            T_Nx4x4
            )
        return d_image.get()

    def construct_detector(self, n_pose):
        n_channels = n_pose
        self.detector = pydrr.Detector(
            pydrr.Detector.make_detector_size(self.image_size, n_channels),
            self.pixel_spacing
            )
        self.projector = pydrr.Projector(self.detector, 1.0).to_gpu()
