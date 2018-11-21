import vtk
import numpy as np

from vtkcast.util import register_numpy_to_vtk, register_vtk_to_numpy
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


@register_vtk_to_numpy(vtk.vtkPolyData)
def convert_poly_vtk_to_numpy(poly, scalar=False):
    """Convert polydata that vtkPolyData convert to numpy array.

    Args:
        poly (vtk.vtkPolyData): input poly data

    Returns:
        Tuple[np.array, np.array]: A pair of vertices and faces
    """

    assert isinstance(poly, vtk.vtkPolyData)

    vertices = vtk_to_numpy(poly.GetPoints().GetData())
    faces = vtk_to_numpy(poly.GetPolys().GetData())
    faces = faces.reshape(-1, 4)[:, 1:]

    if scalar:
        colors = vtk_to_numpy(poly.GetPointData().GetScalars())
        return vertices, faces, colors
    else:
        return vertices, faces


@register_numpy_to_vtk(vtk.vtkPolyData)
def convert_poly_numpy_to_vtk(poly):
    vertices, faces = poly[0:2]
    if not isinstance(faces, vtk.vtkPoints):
        vtkArray = numpy_to_vtk(vertices, deep=1)
        points = vtk.vtkPoints()
        points.SetData(vtkArray)
    else:
        points = vertices

    if not isinstance(faces, vtk.vtkCellArray):
        triangles = vtk.vtkCellArray()

        for i in range(len(faces)):
            triangle = vtk.vtkTriangle()

            triangle.GetPointIds().SetId(0, faces[i, 0])
            triangle.GetPointIds().SetId(1, faces[i, 1])
            triangle.GetPointIds().SetId(2, faces[i, 2])

            triangles.InsertNextCell(triangle)
    else:
        triangles = faces

    # create a polydata object
    polydata = vtk.vtkPolyData()

    # add the geometry and topology to the polydata
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)

    if len(poly) > 2:  # If true, poly has color
        colors = poly[2].astype(np.uint8)
        colors = numpy_to_vtk(colors, deep=2)
        colors.SetName('Colors')
        polydata.GetPointData().SetScalars(colors)

    return polydata
