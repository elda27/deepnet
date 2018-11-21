from deepnet.process.loss.mesh_render import MeshRenderer
import cupy as cp
import numpy as np
import chainer
import vtk
import vtkcast
import imageio
from neural_renderer import get_points_from_angles
import moviepy.editor as mpy
import tqdm


def main():
    renderer = MeshRenderer([(100.0, 91.0, 90.0)])
    renderer.to_gpu()
    renderer.renderer.far = 1e-2
    renderer.renderer.far = 100.0
    # renderer.renderer.to_gpu()

    ply_filename = r"X:\kabashima\ArmReconstruction\dataset-2018-09\poly-dataset-v1\DRF_hmc0002_N\00000008_radius.ply"
    verts, faces = read_ply(ply_filename)
    verts = verts - np.repeat(
        np.mean(verts, axis=0)[np.newaxis], verts.shape[0], axis=0
    )
    verts = normalize_verts(verts)

    verts = chainer.Variable(verts[np.newaxis, ...])
    faces = chainer.Variable(faces[np.newaxis, ...])

    verts.to_gpu()
    faces.to_gpu()

    images = []
    # for i in tqdm.trange(360):
    renderer.views = [(1.0, 0, 90.0)]
    image = renderer(verts, faces)
    image.to_cpu()
    image = (np.squeeze(image.data) * 255).astype(np.uint8)

    imageio.imwrite('image.png', image)

    #    images.append(np.repeat(image[:, :, np.newaxis], 3, axis=2))

    # clip = mpy.ImageSequenceClip(images, fps=10)
    # clip.write_videofile('output.mp4')


def normalize_verts(verts):
    max_xyz = np.amax(np.abs(verts))
    max_xyz = np.array([max_xyz, max_xyz, max_xyz])
    return verts / np.repeat(max_xyz[np.newaxis, ...], verts.shape[0], axis=0)


def read_ply(filename):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(filename)
    reader.Update()
    verts, faces = vtkcast.to_numpy(reader.GetOutput())
    return verts, faces


if __name__ == '__main__':
    main()
