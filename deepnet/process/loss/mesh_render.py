from deepnet.core.registration import register_network
from corenet import declare_node_type
from neural_renderer import Renderer, get_points_from_angles, look_at
import chainer
from chainer.backends import cuda
import chainer.functions as F
import math
import numpy as np
import vtk
import vtkcast


@register_network('network.mesh_renderer')
@declare_node_type('chainer')
class MeshRenderer(chainer.Link):
    def __init__(self, views):
        """Mesh rendering with back propagation

        Args:
            chainer ([type]): [description]
            views (Tuple[float]): View property that is ordered to distance, azimuth, elevation, respectively.
        """

        super().__init__()

        self.views = views

        with self.init_scope():
            self.renderer = Renderer()

    def __call__(self, vertices, faces):
        images = []
        for view in self.views:
            xp = cuda.get_array_module(vertices.data)
            transformed_vertices = look_at(
                vertices,
                get_points_from_angles(*view),
                up=xp.array([0.0, 1.0, 0.0], dtype=xp.float32)
            )
            transformed_vertices.camera_mode = ''
            images.append(
                F.expand_dims(
                    self.renderer.render_silhouettes(
                        transformed_vertices, faces),
                    axis=1
                )
            )

        return F.concat(images, axis=1)
