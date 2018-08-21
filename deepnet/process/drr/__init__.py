from deepnet.core.registration import register_process, add_process
#from . import CpuVolomeProjector
import numpy as np

try:
    from . import GpuVolumeProjector
    VolumeProjector = GpuVolumeProjector
except ModuleNotFoundError:
    pass
except ImportError:
    #VolumeProjector = CpuVolomeProjector
    raise


@register_process()
def HU2Myu(HU_images):
    myu_images = np.fmax((1000.0 + np.float32(HU_images)) * myu_water / 1000.0, 0.0)
    return myu_images

@register_process()
def volume_rendering(
    volume, spacing, case_name = None, 
    hu_volume=True, pose=[], **kwargs
    ):
    assert _ENABLE_PYCUDA_FUNCTION

    if len(pose) == 0:
        pose.append([0, 0, 0, 0, 0, 0])

    projector = GpuVolumeProjector.VolumeProjector(**kwargs)

    cpu_volume = utils.unwrapped(volume) # :TODO: Support cupy to pycuda translation.
    if hu_volume:
        cpu_volume = HU2Myu(cpu_volume, 0.02)

    result = None
    if cpu_volume.ndim >= 4:
        images = []
        for i in range(len(cpu_volume)):
            i_volume = cpu_volume[i]
            images.append(projector(i_volume, spacing[i], case_name[i] if case_name is not None else None, pose))
        result = chainer.Variable(np.transpose(np.array(images), (0, 3, 2, 1)))
    else:
        result = chainer.Variable(projector(cpu_volume, spacing, case_name, pose))

    result.to_gpu()
    result.data.device.synchronize()
    return result
