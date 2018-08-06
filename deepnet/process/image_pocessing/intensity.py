from deepnet.core.registration import register_process
import chainer
from matplotlib.pyplot import get_cmap

@register_process()
def binary(*images, threshold=None):
    assert threshold is not None, '"threshold" is requirement argument'

    bin_images = []
    for image in images:
        assert not isinstance(image, chainer.Variable), 'Unsupported chainer.Variable.'
        if isinstance(image, list):
            bin_images.append([(img > threshold).astype(np.int32) for img in image ])
        else:
            bin_images.append(image > threshold)
    return bin_images

@register_process()
def normalize(*images):
    results = []
    for img in images:
        img = utils.unwrapped(img)
        amax = np.amax(img)
        amin = np.amin(img)
        results.append((img - amax) / (amax - amin + 1e-8))
    return results[0] if len(results) == 1 else results

@register_process()
def blend_image(*image_pairs):
    result_image = []
    for fore, back in zip(image_pairs[::2], image_pairs[1::2]):
        fore = utils.unwrapped(fore)
        back = utils.unwrapped(back)
        if back.shape[1] > fore.shape[1]:
            fore = F.repeat(fore, back.shape[1] // fore.shape[1], axis=1)
        elif back.shape[1] < fore.shape[1]:
            back = F.repeat(back, fore.shape[1] // back.shape[1], axis=1)
        result_image.append(normalize(fore) * 0.5 + normalize(back) * 0.5)
    return result_image

@register_process()
def diff_image(*input, absolute=False):
    output = []
    abs_method = None
    if absolute:
        abs_method = F.absolute
    else:
        abs_method = lambda x: x

    for i, j in zip(input[::2], input[1::2]):
        output.append(abs_method(i - j))
    return output

