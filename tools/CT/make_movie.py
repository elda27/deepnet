import cv2
import mhd
import glob
import os.path
import imgproc
import numpy as np
import pandas as pd
import tqdm
from scipy import ndimage
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-s', '--shape', default=(128, 128), nargs=2)
    
    root_dir = args.input
    block_shape = tuple(args.shape)
    downsample_shape = np.array((1,) + block_shape[::-1])
    frame_rate = 10
    n_column = 8
    
    volumes = []
    infos = {}

    print('Load images')
    glob_images = glob.glob(os.path.join(root_dir, '*.mhd'))
    for image_filename in tqdm.tqdm(list(glob_images)):
        image, header = mhd.read(image_filename)
        label, _ = os.path.splitext(os.path.basename(image_filename))
        
        # Set info
        infos.setdefault('Label', []).append(label)

        infos.setdefault('Dim X', []).append(image.shape[2])
        infos.setdefault('Dim Y', []).append(image.shape[1])
        infos.setdefault('Dim Z', []).append(image.shape[0])

        infos.setdefault('Spacing X', []).append(header['ElementSpacing'][0])
        infos.setdefault('Spacing Y', []).append(header['ElementSpacing'][1])
        infos.setdefault('Spacing Z', []).append(header['ElementSpacing'][2])

        infos.setdefault('Min intensity', []).append(np.amin(image))
    
        # Set volume
        scale = downsample_shape / np.array(image.shape)
        volumes.append(ndimage.zoom(image, scale, mode='nearest'))

    pd.DataFrame(infos).to_csv('info.csv')

    min_intensity = min( [ np.amin(volume) for volume in volumes ] )
    max_intensity = max( [ np.amax(volume) for volume in volumes ] )
    max_slice = max( [ volume.shape[0] for volume in volumes ] )

    print('Preprocess images.')
    image_list = [ [] for _ in range(max_slice) ]
    for volume in tqdm.tqdm(volumes):
        for i in tqdm.trange(max_slice):
            if i >= volume.shape[0]:
                image_list[i].append(np.zeros(block_shape, dtype=np.uint8))
                continue
            normalized_image = ((volume[i] - min_intensity) / (max_intensity - min_intensity)).astype(np.uint8)
            image_list[i].append(normalized_image)

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    writer = None

    print('Write video.')
    for images in tqdm.tqdm(image_list):
        img = imgproc.make_tile_2d(images, (None, n_column))
        
        if writer is None:
            writer = cv2.VideoWriter('video.mp4', fourcc, frame_rate, img.shape)
        writer.write(img)

    writer.release()


if __name__ == '__main__':
    main()
