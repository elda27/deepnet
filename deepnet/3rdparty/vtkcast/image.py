import vtk
import numpy as np
from vtkcast.util import register_numpy_to_vtk, register_vtk_to_numpy, \
    dtype_vtk_to_numpy, dtype_numpy_to_vtk

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


@register_vtk_to_numpy(vtk.vtkImageData)
def convert_vtk_to_numpy(image):
    shape = image.GetDimensions()
    ch = image.GetNumberOfScalarComponents()
    scalar_pointer = image.GetPointData().GetScalars()
    a = vtk_to_numpy(scalar_pointer)
    return np.squeeze(a.reshape(shape + (ch,)))


@register_numpy_to_vtk(vtk.vtkImageData)
def convert_numpy_to_vtk(image):
    image_data = vtk.vtkImageData()
    depth_array = numpy_to_vtk(
        image.ravel(),
        deep=True,
        array_type=dtype_numpy_to_vtk(image.dtype)
    )

    image_data.SetDimensions(image.shape)
    image_data.SetSpacing([1.0] * image.ndim)
    image_data.SetOrigin([0.0] * image.ndim)
    image_data.GetPointData().SetScalars(depth_array)

    return image_data
