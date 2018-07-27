import auto_path
import deepnet

import argparse

import os
import os.path

import sys

import pandas as pd

import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input configuration file')
    parser.add_argument('-o', '--output', nargs='+', type=str, required=True, help='Output filename')
    parser.add_argument('--image', type=str, nargs='+', required=True, help='Image name')
    parser.add_argument('--spacing', nargs='+', type=str, required=True, help='Output filename')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--mode', type=str, default='train', help='Mode of dataset construction')
    parser.add_argument('--disable-extension', action='store_true')
    args = parser.parse_args()

    config = deepnet.config.load(args.input)
    dataset = None
    num_items = 0
    if args.disable_extension:
        dataset = deepnet.utils.dataset.GeneralDataset(config, None)
        num_items = len(dataset)
    else:
        dataset = deepnet.utils.dataset.CachedDataset(config, None, args.mode)
        num_items = super(deepnet.utils.dataset.CachedDataset, dataset).__len__()

    informations = dict()

    os.makedirs(args.output_dir, exist_ok=True)

    for i in tqdm.tqdm(list(range(0, num_items))):
        dataset.cache = None
        if args.disable_extension:
            process_example(dataset, i, args, informations)
        else:
            dataset.construct_cache([i])
            for j in tqdm.trange(len(dataset)):
                process_example(dataset, j, args, informations)
    df = pd.DataFrame.from_dict(informations, orient='columns')
    df.to_excel('info.xlsx')

def process_example(dataset, i, args, informations):
    data = dataset.get_example(i)
    for image_name, spacing_name, output_filename in zip(args.image, args.spacing, args.output):
        image = data[0][image_name]
        spacing = data[0][spacing_name]
        case_name = data[0]['case_name']
        deepnet.utils.visualizer.save_image(
            os.path.join(args.output_dir, output_filename % (case_name, i)),
            image, spacing
            )
        write_information(informations, image_name, image, spacing)



def write_information(info, image_name, image, spacing):
    info.setdefault(image_name + '/Dim X', []).append(image.shape[-1])
    info.setdefault(image_name + '/Dim Y', []).append(image.shape[-2])
    if image.ndim > 2:
        info.setdefault(image_name + '/Dim Z', []).append(image.shape[-3])

    info.setdefault(image_name + '/Spacing X', []).append(spacing[0])
    info.setdefault(image_name + '/Spacing Y', []).append(spacing[1])
    if image.ndim > 2:
        info.setdefault(image_name + '/Spacing Z', []).append(spacing[2])
    return info

if __name__ == '__main__':
    main()

