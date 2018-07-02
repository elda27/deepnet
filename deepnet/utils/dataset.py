import chainer
import os.path
import glob
from . import mhd
import warnings
import imageio
import numpy as np

class XpDataset(chainer.dataset.DatasetMixin):
    image_format = dict(
        default='*_image.mhd',
        lcn='*_image_lcn.mhd',
        gcn='*_image_gcn.mhd',
    )
    label_format = '*_label.mhd'

    def __init__(self, patient_root, case_names, image_type='default', image=True, label=True):
        assert image or label, 'Unused whole images.'
        cls = XpDataset
        if image_type not in cls.image_format:
            warnings.warn('Unknown format: {}\n will use default image'.format(image_type))
            image_type = 'default'

        self.case_names = case_names
        self.use_image = image
        self.use_label = label
        self.image_paths = []
        self.label_paths = []
        self.case_names = []
        for case_name in case_names: # Set case names
            image_glob = os.path.join(patient_root, case_name, cls.image_format[image_type])
            self.image_paths.extend(glob.glob(image_glob))
            label_glob = os.path.join(patient_root, case_name, cls.label_format)
            self.label_paths.extend(glob.glob(label_glob))
            self.case_names.extend([ case_name for i in range(max(len(self.image_paths), len(self.label_paths)) - len(self.case_names)) ])
        assert not self.use_image or len(self.image_paths) != 0, 'This dataset is empty image. (root_path:{})'.format(patient_root)
        assert not self.use_label or len(self.label_paths) != 0, 'This dataset is empty label. (root_path:{})'.format(patient_root)
        assert (not (image and label)) or len(self.image_paths) == len(self.label_paths), \
            "Unmatched to count of image and label file. (image:{}, label:{})".format(len(self.image_paths), len(self.label_paths))

    def __len__(self):
        return len(self.image_paths) if len(self.image_paths) != 0 else len(self.label_paths)

    def get_example(self, index):
        result = dict()
        if self.use_label: # Load label
            label, header = mhd.read(self.label_paths[index])
            spacing = header['ElementSpacing']
            result['label'] = label
            result['spacing'] = spacing

        if self.use_image: # Load image
            image, header = mhd.read(self.image_paths[index])
            if image.ndim < 3:
                image = np.expand_dims(image, axis=0)
            spacing = header['ElementSpacing']
            result['image'] = image
            result['spacing'] = spacing
        
        result['case_name'] = self.case_names[index]

        return [ result ]

class InstantDataset(chainer.dataset.DatasetMixin):
    def __init__(self, image_root, use_ratio, use_backward=False):
        self.illust_images =  list(glob.glob(os.path.join(image_root, 'illust-db', '*.jpg')))
        if use_backward:
            self.illust_images = self.illust_images[int((1 - use_ratio) * len(self.illust_images)):]
        else:
            self.illust_images = self.illust_images[:int(use_ratio * len(self.illust_images))]

    def __len__(self):
        return len(self.illust_images)
        #return max((len(self.image_paths), len(self.illust_images)))

    def get_example(self, index):
        illust = imageio.imread(self.illust_images[index]).astype(np.float32)
        if illust is None or illust.ndim == 1 or illust.ndim == 0:
            illust = imageio.imread(self.illust_images[index - 1]).astype(np.float32)
        if illust.ndim == 2:
            illust = np.expand_dims(illust, axis=2)
        illust = np.transpose(illust, (2, 1, 0)).astype(np.float32)
        result = dict(
            label=illust,
        )
        return [ result ]

class RangeArray:
    def __init__(self):
        self.values = []
        self.length = 0
    
    def append(self, begin, end, value):
        self.values.append((begin, end, value))
        self.length = max(self.length, end)

    def get(self, index):
        """ Get value from index.
        
        Args:
            index (int): list index
        
        Raises:
            IndexError: 
        
        Returns:
            any: List value
        
        Warnings:
            This function can't receive slice object.
            If you use slice, you use __getitem__ method.
        """
        
        for begin, end, value in self.values:
            if begin <= index < end:
                return value
        raise IndexError('list idnex out of range')

    def __getitem__(self, index):        
        if isinstance(index, slice):
            values =[]
            for i in range(*index.indices()):
                values.append(self.get(i))
            return tuple(values)
        elif isinstance(index, int):
            return self.get(values)
        else:
            raise TypeError

    def __len__(self):
        return self.length

class GeneralDataset(chainer.dataset.DatasetMixin):
    input_methods = {}
    used_indices = 0.0

    def __init__(self, config, indices):
        DEFAULT_GROUP = -1
        groups = {}
        for input_field in config['input']:
            if 'stage' not in input_field:
                groups.setdefault(DEFAULT_GROUP, []).append(input_field)
            else:
                groups.setdefault(input_field['stage'], []).append(input_field)
        
        self.case_names = config['config']['case_names'] if 'case_names' in config['config'] else None 
        self.stage_inputs = []

        for _, inputs in sorted(groups.items(), key=lambda x:x[0]):
            stage_input = {}
            for input_ in inputs:
                input_['type']
                
                case_names = None

                paths = []
                if not isinstance(indices, float) and indices is not None:
                    # If dataset has some cases
                    case_names = RangeArray()

                    for case_name in indices:  # Replace case_name 
                        assert case_name in self.case_names, 'Unknown case name: ' + case_name + str(self.case_names)
                        current_paths = self.glob_case_dir(input_['paths'], '<case_names>', case_name)
                        case_names.append(len(paths), len(paths) + len(current_paths), case_name)
                        paths.extend(current_paths)
                else:
                    assert GeneralDataset.used_indices < 1.0, 'Failed to split dataset because the dataset is fully used.'
                    for path in input_['paths']:
                        paths.extend(glob.glob(path))

                    start_ratio = GeneralDataset.used_indices

                    if indices is None or (start_ratio + indices) >= 1.0:
                        indices = 1.0
                        GeneralDataset.used_indices = 1.0
                    else:
                        GeneralDataset.used_indices += indices

                    paths = paths[int(len(paths) * start_ratio) : int(len(paths) * GeneralDataset.used_indices)]

                assert len(paths), 'Founds dataset is empty. ' + str(input_['paths'])
                stage_input[tuple(input_['label'])] = {
                    'input': load_image,
                    'paths': paths,
                }
                
                if 'case_name' in stage_input and case_names is not None:
                    stage_input['case_name'] = case_names
                
            self.stage_inputs.append(stage_input)

    def get_example(self, index):
        inputs = []
        for stage_input_method in self.stage_inputs:
            stage_input = {}
            for label, input_method in stage_input_method.items():
                paths = input_method['paths']
                data = input_method['input'](paths[index % len(paths)])
                if isinstance(label, tuple):
                    stage_input.update(zip(label, data))
                else:
                    stage_input[label] = data
            inputs.append(stage_input)
        return inputs

    def glob_case_dir(self, path_strs, var_name, var_value):
        paths = []
        for path in path_strs:
            path_list = list(glob.glob(path.replace(var_name, var_value)))
            paths.extend(path_list)
        return paths

    def __len__(self):
        return max((len(data['paths']) for stage_input in self.stage_inputs for data in stage_input.values()))

def register_input_method(type_):
    def _register_input_method(func):
        GeneralDataset.input_methods[type_] = func
        return func
    return _register_input_method

@register_input_method('image')
def load_image(filename):
    _, ext = os.path.splitext( os.path.basename(filename) )

    if ext in ('.mha', '.mhd'):
        [img, img_header] = mhd.read(filename)
        spacing = img_header['ElementSpacing']
        
        return img, spacing
        
    elif ext in ('.png', '.jpg'):
        img = imageio.imread(filename)
        return np.transpose(img.astype(np.float32), (2, 1, 0))
    raise NotImplementedError('Not implemented extension: (File: {}, Ext: {})'.format(filename, ext))