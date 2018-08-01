#from . import CpuVolomeProjector
import pydrr.utils

try:
    from . import GpuVolomeProjector
    VolumeProjector = GpuVolomeProjector
except ImportError:
    #VolumeProjector = CpuVolomeProjector
    raise
