from deepnet.core.registration import register_network
from corenet import declare_node_type
from neural_renderer import Renderer, get_points_from_angles
import chainer
import chainer.functions as F


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
            self.renderer.eye = get_points_from_angles(*view)
            images.append(
                F.expand_dims(
                    self.renderer.render_silhouettes(vertices, faces),
                    axis=1
                )
            )

        return F.concat(images, axis=1)
