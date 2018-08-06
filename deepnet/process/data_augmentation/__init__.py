from deepnet.core.registration import register_process
from . import image_augmentation, volume_augmentation

@register_process()
def random_transform(
    *input,
    mode='image',
    rotation=0.0,    # [deg]
    translation=0.0, # [%]
    zoom=0.0,        # [%]
    intensity=0.0   # [intensity]
    ):

    outputs = []
    for image in input:
        outputs.append(augmentations[mode](
            image,
            rotation, translation, zoom,
            intensity, 
        ))
    return outputs

augmentations = {}
def register_augmentation(name):
    def _register_augmentation(func):
        augmentations[name] = func
        return func
    return _register_augmentation

@register_augmentation('image')
def augmentation_2d(
    images,
    rotation_range=0.0,    # [deg]
    translation_range=0.0, # [%]
    zoom_range=0.0,        # [%]
    intensity_range=0.0,   # [intensity]
):
    # x is a single image, so it doesn't have image number at index 0
    generator = image_augmentation.ImageDataGenerator(
        rotation_range   = rotation_range,
        translation_range= translation_range,
        zoom_range     = zoom_range,
        intensity_range= intensity_range,
    )
    output = []
    image_iter = iter(images)
    output.append(generator.random_transform(next(image_iter)))
    for x in image_iter:
        output.append(generator.fixed_transform(x))
    return output

@register_augmentation('volume')
def augmentation_3d(
    images,
    rotation_range=0.0,    # [deg]
    translation_range=0.0, # [%]
    zoom_range=0.0,        # [%]
    intensity_range=0.0,   # [intensity]
):
    # x is a single image, so it doesn't have image number at index 0
    generator = volume_augmentation.VolumeDataGenerator(
        rotation_range   = rotation_range,
        translation_range= translation_range,
        zoom_range     = zoom_range,
        intensity_range= intensity_range,
    )
    output = []
    image_iter = iter(images)
    output.append(generator.random_transform(next(image_iter)))
    for x in image_iter:
        output.append(generator.fixed_transform(x))
    return output



