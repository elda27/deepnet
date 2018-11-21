import vtk
import numpy as np

from vtkcast.util import register_numpy_to_vtk, register_vtk_to_numpy
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


@register_vtk_to_numpy(vtk.vtkMatrix4x4)
def convert_matrix4x4_vtk_to_numpy(matrix):
    temp = [0] * 16
    matrix.DeepCopy(temp, matrix)
    return np.asarray(temp).reshape((4, 4))


@register_numpy_to_vtk(vtk.vtkMatrix4x4)
def convert_matrix4x4_numpy_to_vtk(matrix):
    vtk_matrix = vtk.vtkMatrix4x4()
    vtk_matrix.DeepCopy(matrix.ravel().tolist())
    return vtk_matrix
