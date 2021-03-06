from deepnet.core.registration import register_process, add_process
from .CpuVolumeProjector import CpuVolumeProjector
import numpy as np

try:
    from .GpuVolumeProjector import GpuVolumeProjector
    VolumeProjector = GpuVolumeProjector
except ImportError:
    VolumeProjector = CpuVolumeProjector
    pass


@register_process()
def HU2Myu(HU_images, myu_water):
    myu_images = np.fmax((1000.0 + np.float32(HU_images))
                         * myu_water / 1000.0, 0.0)
    return myu_images


@register_process()
def volume_rendering(
    volume, spacing, case_name=None,
    hu_volume=True, pose=[], **kwargs
):
    assert _ENABLE_PYCUDA_FUNCTION

    if len(pose) == 0:
        pose.append([0, 0, 0, 0, 0, 0])

    projector = VolumeProjector(**kwargs)

    # :TODO: Support cupy to pycuda translation.
    cpu_volume = utils.unwrapped(volume)
    if hu_volume:
        cpu_volume = HU2Myu(cpu_volume, 0.02)

    result = None
    if cpu_volume.ndim >= 4:
        images = []
        for i in range(len(cpu_volume)):
            i_volume = cpu_volume[i]
            images.append(projector(
                i_volume, spacing[i], case_name[i] if case_name is not None else None, pose))
        result = chainer.Variable(np.transpose(np.array(images), (0, 3, 2, 1)))
    else:
        result = chainer.Variable(
            projector(cpu_volume, spacing, case_name, pose))

    result.to_gpu()
    result.data.device.synchronize()
    return result
