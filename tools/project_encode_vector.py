import numpy as np
import sys
import argparse
import deepnet
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import chainer

pallete = [
    (233, 66, 59),   # Red
    (222, 29, 98),   # Pink
    (150, 41, 72),   # Purple
    (101, 59, 79),   # Deep purple
    (68, 82, 177),   # Indigo
    (65, 151, 239),  # Blue
    (62, 170, 241),  # Light blue
    (66, 188, 211),  # Cyan
    (50, 150, 136),  # Teal
    (91, 175, 87),   # Green
    (145, 195, 85),  # Light green
    (208, 220, 78),  # Lime
    (252, 235, 83),  # Yellow
    (249, 193, 51),  # Amber
    (248, 151, 40),  # Orange
    (244, 86, 45),   # Deep Orange
    (117, 85, 73),   # Brown
    (158, 158, 158), # Gray
    (99, 125, 138),  # Blue Gray
]

pallete =  [
    'red',
    'slateblue',
    'purple',
    'deeppink', 
    'indigo',
    'blue',
    'olive',
    'cyan',
    'teal',
    'green',
    'lightgreen',
    'lime',
    'yellow',
    'amber',
    'orange',
    'maroon',
    'brown',
    'gray',
]

def main():
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--vector', type=str, required=True, help='Encoded vectors')
    parser.add_argument('--dataset-dir', type=str, default=None, required=False, help='Dataset directory')
    parser.add_argument('--output-dir', type=str, required=False, help='Output directory')
    parser.add_argument('--index', type=str, required=True, help='index file path')

    if len(sys.argv) < 2:
        parser.print_help()
        return 

    args = parser.parse_args()

    index = deepnet.utils.parse_index_file(args.index)

    encoded_vector = np.load(args.vector)
    if isinstance(encoded_vector[0, 0], chainer.Variable):
        for i in range(encoded_vector.shape[1]):
            for j in range(encoded_vector.shape[0]):
                encoded_vector[j, i] = np.float32(encoded_vector[j, i].data)

    encoded_vector = np.ascontiguousarray(encoded_vector, dtype=np.float32)


    case_names = deepnet.utils.parse_index_file(args.index)
    #colors = sum([ [ float(i) / len(index) ] * num_sample_in_case for i, name in enumerate(index) ], [])

    for i in range(encoded_vector.shape[1]):
        plt.clf()
        plot_histogram(encoded_vector, index, i)
        plt.savefig('encoded_bar_{}.png'.format(i))
        plt.savefig('encoded_bar_{}.pdf'.format(i))
    #plot_boxplot(encoded_vector, index)
    #plot_tSNE(encoded_vector, index)
    #plt.show()

    if args.dataset_dir is not None:
        dataset = deepnet.utils.dataset.XpDataset(args.dataset_dir, case_names, image=False, label=True)

def plot_boxplot(encoded_vector, index, dim=0):
    plt.boxplot(encoded_vector)
    
def plot_histogram(encoded_vector, index, dim=0):
    plt.hist(encoded_vector[:, dim], bins=20, range=(-1.0, 1.0), color='blue', edgecolor='black', linewidth=1.2)

def plot_tSNE(encoded_vector, index, dim=None):
    num_sample_in_case = int(encoded_vector.shape[0] // len(index))

    reduced_vector = TSNE(n_components=2, random_state=0).fit_transform(encoded_vector)
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i, case_name in enumerate(index):
        ax.scatter(
            reduced_vector[num_sample_in_case * i : num_sample_in_case * (i + 1), 0], 
            reduced_vector[num_sample_in_case * i : num_sample_in_case * (i + 1), 1],
            c=pallete[i],
            label=case_name,
            alpha=0.3, edgecolors=pallete[i]
            )
    plt.legend()


if __name__ == '__main__':
    main()