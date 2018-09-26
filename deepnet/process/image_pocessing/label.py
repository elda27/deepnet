from deepnet.core.registration import register_process
from matplotlib.pyplot import get_cmap

@register_process()
def make_overlap_label(*images):
    cmap = get_cmap(color)
    result_image = []
    for image in images:
        img = utils.unwrapped(image)
        
        index_img = np.argmax(
            np.concatenate((np.ones( (img.shape[0], 1) + img.shape[2:], dtype=img.dtype) * 1e-1, img), axis=1), 
            axis=1
            )
        result_image.append(map_index_label(index_img))
    return result_image

@register_process()
def map_index_label(label):
    """Mapping color to index image
    
    Args:
        label (images): Input images aligned by (N, 1, S, ...) or 
                        (N, S, ...) (N means N sample so it is same as batch size, 
                        S mean shape of image).
        color (list, optional): color of input images, Defaults to colors.
    
    Returns:
        numpy.ndarray: color images (N, 3, S, ...), S is same as input image.
    """

    color = 'tab10'
    fold  = 10

    indices = list(np.unique(label))
    indices.remove(0)
    color_image = np.tile(
        np.expand_dims(
            np.zeros_like(label), 
            axis=label.ndim
            ), 
        (1,) * label.ndim + (3,)
    )

    for i in reversed(indices):
        color = cmap(((i - 1) % fold) / fold)
        mask = label == i
        r = np.expand_dims(mask * color[0] * 255, axis=mask.ndim)
        g = np.expand_dims(mask * color[1] * 255, axis=mask.ndim)
        b = np.expand_dims(mask * color[2] * 255, axis=mask.ndim)
        color_mask = np.tile(
            np.expand_dims(
                np.logical_not(mask), 
                axis=mask.ndim
                ), 
            (1,) * mask.ndim + (3,)
        )
        color_image = color_image * color_mask * np.concatenate((r, g, b), axis=mask.ndim)
    
    return np.transpose(color_image, (1, 0, 2, 3))

