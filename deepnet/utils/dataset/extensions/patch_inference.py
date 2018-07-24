from ..dataset import DatasetExtension, register_extension
import random
import chainer
from functools import reduce

@register_extension('patch_inference')
class PatchBasedInference(DatasetExtension):
    def __init__(self, config):
        self.patch_shape = config['patch_shape']
        self.ndim = len(self.patch_shape)

        self.cache_size = config.get('cache_size', 50)
        self.population_size = self.cache_size * 10
        self.foreground_ratio = config.get('foreground_ratio', 0.6)
        self.background_pixel = config.get('background_pixel', -1000)
        
        for target in config['targets']:
            self.source = target['source']
            if 'destination'in target:
                destination = target['destination']
                if isinstance(destination, dict):
                    self.destination = destination['image']
                    self.destination_patch_region = destination.get('patch_region', destination + '.patch_region')
                else:
                    self.destination = destination
                    self.destination_patch_region = destination + '.patch_region'
            else:
                self.destination = self.source + '.patch'
                self.destination_patch_region = self.source + '.patch_region'

    def __call__(self, stage_input):
        if self.source not in stage_input:
            return

        cropper = None
        if self.is_train():
            cropper = RandomCropper(
                self.patch_shape, 
                n_samples=self.population_size, 
                fore_prob=self.foreground_ratio,
                back_pix =self.background_pixel,
                n_caches =self.cache_size,
                )
        else:
            cropper = PatchCropper(self.patch_shape)
        
        patches, patch_regions = cropper(stage_input[self.source])
        #assert all([ patch.ndim == len(self.patch_shape) for patch in patches ] ), 'Failed to crop image.'

        results = [ { self.destination: p, self.destination_patch_region: pr } for p, pr in zip(patches, patch_regions) ]

        return results

class RandomCropper:
    def __init__(self, patch_shape, n_samples = 100, fore_prob=0.6, back_pix = -1000, use_cache = True, n_caches = 50):
        """Random crop for patch training.
        
        Args:
            patch_shape (tuple[int]): Patch shape
            n_samples (int, optional): Defaults to 100. A number of samples for cache generation.
            fore_prob (float, optional): Defaults to 0.6. 
                                         A ratio of foreground images in cache images.
            back_pix (int, optional): Defaults to -1000. 
                                      The background pixel value to compute foreground area.
            use_cache (bool, optional): Defaults to True. To use cache image. This parameter always True.
            n_caches (int, optional): Defaults to 50. A number of cache size.
        """

        self.patch_shape = patch_shape
        self.n_samples = n_samples
        self.fore_prob = fore_prob
        self.back_pix  = back_pix
        self.use_cache = use_cache
        self.n_caches = n_caches
        self.cache = {}

    def __call__(self, image):
        crop_images = []
        result_images = []
        result_crop_regions = []

        for i in range(self.n_samples):
            crop_image, crop_region = self.random_crop(image)
            xp = chainer.cuda.get_array_module(crop_image)
            area = xp.sum(crop_image != self.back_pix)
            
            crop_images.append( (area, crop_image, crop_region) )
        
        crop_images = sorted(crop_images, key=lambda x: x[0], reverse=True)
        fore_end_of_index = int(self.n_caches * self.fore_prob)
        result_images.extend([ crop_images[i][1] for i in range(fore_end_of_index) ])
        result_crop_regions.extend(crop_images[:fore_end_of_index][2])

        random_choice = random.choices(crop_images[fore_end_of_index:], k = self.n_caches - fore_end_of_index)
        result_images.extend([ elem[1] for elem in random_choice ])
        result_crop_regions.extend([ elem[2] for elem in random_choice ])

        return result_images, result_crop_regions

    def random_crop(self, image):
        crop_start = [ int(min(random.random() * s, s - ps)) for s, ps in zip(image.shape, self.patch_shape) ]
        crop_end = [ s + ps for s, ps in zip(crop_start, self.patch_shape) ]
        slices = tuple(map(slice, crop_start, crop_end))
        return image[slices], tuple(zip(crop_start, crop_end))

class PatchCropper:
    def __init__(self, patch_shape):
        self.patch_shape = patch_shape

    def __call__(self, array):
  
        index = 0
        max_indices = []
        for s, ps in zip(array.shape, self.patch_shape):
            max_indices.append(int(s // ps))

        result_images = []
        result_crop_regions = []
        for index in range(reduce(lambda x, y: x * y, max_indices)):
            indices = []
            ci = 1
            for mi in max_indices:
                indices.append(index / ci % mi)
                ci *= mi

            start = [ int(min(int(index * ps), s - ps)) for s, ps in zip(array.shape, self.patch_shape) ]
            end = [ s + ps for s, ps in zip(start, self.patch_shape) ]
            slices = tuple(map(slice, start, end))

            result_images.append(array[slices])
            result_crop_regions.append(tuple(zip(start, end)))

        return result_images, result_crop_regions
