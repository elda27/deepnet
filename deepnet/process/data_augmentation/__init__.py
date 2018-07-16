from . import image_augmentation

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



