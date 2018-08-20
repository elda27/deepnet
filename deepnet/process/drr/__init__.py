#from . import CpuVolomeProjector
import pydrr.utils

try:
    from .GpuVolumeProjector import GpuVolumeProjector
    VolumeProjector = GpuVolumeProjector
except ImportError:
    #VolumeProjector = CpuVolomeProjector
    raise
