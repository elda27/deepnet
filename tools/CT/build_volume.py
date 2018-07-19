import mhd
import numpy as np
import glob
import tqdm
import argparse
import os.path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--z-slice', type=float, default=0.625)
    parser.add_argument('-i', '--input', type=str, required=True)
    args = parser.parse_args()

    for patient_dir in tqdm.tqdm(glob.glob(os.path.join(args.input, '*'))):
        patient_id = os.path.basename(patient_dir)
        image_glob = glob.glob(os.path.join(patient_dir, '*.mhd'))
        if len(image_glob) == 0:
            continue

        spacing = None
        volume = None
        for i, slice_file in enumerate(tqdm.tqdm(image_glob)):
            image, header = mhd.read(slice_file)
            slice_image = np.squeeze(image)
            if spacing is None:
                spacing = header['ElementSpacing'][:2] + [args.z_slice, ]
                volume = np.zeros( (len(image_glob), ) + slice_image.shape, dtype=np.float32 )
            volume[i] = slice_image

        mhd.write(patient_id + '.mhd', volume, {'ElementSpacing': spacing})
    

if __name__ == '__main__':
    main()
