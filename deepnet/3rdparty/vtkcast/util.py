import vtk
import numpy

_registered_vtk_to_numpy = {}
_registered_numpy_to_vtk = {}


def register_vtk_to_numpy(type_):
    def _register_vtk_to_numpy(func):
        _registered_vtk_to_numpy[type_] = func
        return func
    return _register_vtk_to_numpy


def register_numpy_to_vtk(type_):
    def _register_vtk_to_numpy(func):
        _registered_numpy_to_vtk[type_] = func
        return func
    return _register_vtk_to_numpy


def to_numpy(vtk_object, *args, **kwargs):
    return _registered_vtk_to_numpy[type(vtk_object)](vtk_object, *args, **kwargs)


def to_vtk(vtk_type, numpy_array, *args, **kwargs):
    return _registered_numpy_to_vtk[vtk_type](numpy_array, *args, **kwargs)


_depth_vtk_to_numpy = {
    vtk.VTK_CHAR: numpy.int8,
    vtk.VTK_UNSIGNED_CHAR: numpy.uint8,
    vtk.VTK_SHORT: numpy.int16,
    vtk.VTK_UNSIGNED_SHORT: numpy.uint16,
    vtk.VTK_INT: numpy.int32,
    vtk.VTK_UNSIGNED_INT: numpy.uint32,
    vtk.VTK_FLOAT: numpy.float32,
    vtk.VTK_DOUBLE: numpy.float64,
}


def dtype_vtk_to_numpy(depth):
    """VTK depth to numpy depth.

    Args:
        depth : vtk depth value

    Returns:
        numpy.depth: converted depth type.
    """

    return _depth_vtk_to_numpy[depth]


_depth_numpy_to_vtk = {
    numpy.dtype(numpy.int8).name: vtk.VTK_CHAR,
    numpy.dtype(numpy.uint8).name: vtk.VTK_UNSIGNED_CHAR,
    numpy.dtype(numpy.int16).name: vtk.VTK_SHORT,
    numpy.dtype(numpy.uint16).name: vtk.VTK_UNSIGNED_SHORT,
    numpy.dtype(numpy.int32).name: vtk.VTK_INT,
    numpy.dtype(numpy.uint32).name: vtk.VTK_UNSIGNED_INT,
    numpy.dtype(numpy.float32).name: vtk.VTK_FLOAT,
    numpy.dtype(numpy.float64).name: vtk.VTK_DOUBLE,
}


def dtype_numpy_to_vtk(depth):
    """VTK depth to numpy depth.

    Args:
        depth : vtk depth value

    Returns:
        numpy.depth: converted depth type.
    """

    return _depth_numpy_to_vtk[numpy.dtype(depth).name]
