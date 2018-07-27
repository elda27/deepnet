from deepnet.utils.postprocess import base
from deepnet import utils
import numpy as np

@base.register_postprocess('concat_patch')
class PatchConcat(base.PostProcessor):
    def __init__(self, 
        patch, 
        crop_region, 
        case_name,
        source_volume = None, 
        image_shape = None,
        output_root = None,
        ):
        super().__init__(base.PostProcessTrigger.AfterEachProcess, base.PostProcessTrigger.AfterProcess)

        self.patch = patch
        self.case_name = case_name
        self.crop_region = crop_region
        self.image_shape = image_shape
        self.source_volume = source_volume
        self.current_image = None
        self.images = {}
        self.output_root = 'patch_concat' if output_root is None else output_root
        
    def peek(self, variable):
        patch_image = variable[self.patch]
        case_name   = variable[self.case_name]
        crop_region = variable[self.crop_region]
        source_volume = variable[self.source_volume] if self.source_volume is not None else [None] * patch_image

        for patch, case_name, crop_region, source_volume in zip(patch_image, case_name, crop_region, source_volume):
            if case_name not in self.images:
                image_shape = self.image_shape if self.image_shape is not None else source_volume.shape
                self.images[case_name] = np.zeros(image_shape, np.float32)
            self.images[case_name][tuple(( slice(*c) for c in crop_region ))] = utils.unwrapped(patch)

    def get_result(self):
        variables = {}
        for case_name, image in self.images.items():
            variables.setdefault(self.output_root + '/image', []).append(image)
            variables.setdefault(self.output_root + '/case_name', []).append(case_name)

        return variables