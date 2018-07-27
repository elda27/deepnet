import chainer
import os.path
import glob
from deepnet.utils import mhd
import warnings
import imageio
import numpy as np
import abc

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
            return self.get(index)
        elif np.issubdtype(index, np.integer):
            return self.get(index)
        else:
            raise TypeError('Unsupported index type. Actual type:{}, Value: {}'.format(type(index), index) )

    def __len__(self):
        return self.length

class GeneralDataset(chainer.dataset.DatasetMixin):
    input_methods = {}
    extensions    = {}
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
        self.cache = []
        self.iteration = 1000

        if indices is None:
            indices = self.case_names

        for _, inputs in sorted(groups.items(), key=lambda x:x[0]):
            stage_input = {}
            for input_ in inputs:
                input_['type']
                
                case_names = None

                paths = []
                if not isinstance(indices, float) and indices is not None:
                    # If dataset has some cases
                    case_names = []

                    for case_name in indices:  # Replace case_name 
                        assert case_name in self.case_names, 'Unknown case name: ' + case_name + str(self.case_names)
                        current_paths = self.glob_case_dir(input_['paths'], '<case_names>', case_name)
                        case_names.extend([case_name] * len(current_paths))
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

                labels = tuple(input_['label'])
                stage_input[labels] = {
                    'input': load_image,
                    'paths': paths,
                }
                
                if case_names is not None:
                    stage_input[labels]['case_name'] = case_names
                
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

                if 'case_name' in input_method:
                    stage_input['case_name'] = input_method['case_name'][index]

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

    @staticmethod
    def generate_extension(extension_field):
        cls = GeneralDataset
        return cls.extensions[extension_field['type']](extension_field)

class CachedDataset(GeneralDataset):
    def __init__(self, config, indices, mode, construct_cache = True):
        super().__init__(config, indices)
        self.mode = mode

        self.extensions = []
        for extension_field in config['extension']:
            extension = GeneralDataset.generate_extension(extension_field)
            extension.set_mode(self.mode)
            self.extensions.append(extension)

        self.cache = None

        if construct_cache:
            self.construct_cache()

    def construct_cache(self, iter_range = None):
        if iter_range is None:
            iter_range = range(super().__len__())
            
        for i in iter_range:
            stage_input = super().get_example(i)
            if self.cache is None:
                self.cache = [ [] for _ in stage_input ]
            
            for j in range(len(stage_input)):
                self.cache[j].extend(self.process_extension([ stage_input[j] ]))
            
    def process_extension(self, data_list, extensions = None):
        if extensions is None:
            extensions = list(reversed(self.extensions))

        result_data = []
        extension = extensions[-1]
        for data in data_list:
            gen_data = extension(data)
            
            if isinstance(gen_data, list):
                for d in gen_data:
                    d.update({ key: value for key, value in data.items() if key not in gen_data })
                    result_data.append(d)
            else:
                result_data.append(data.update(gen_data))
    
        extensions.pop()
        if len(extensions) == 0:
            return result_data

        return self.process_extension(result_data, extensions)

    def get_example(self, index):
        return [ self.cache[i][index] for i in range(len(self.cache)) ]

    def __len__(self):
        return max((len(stage_input) for stage_input in self.cache))

class DatasetExtension(metaclass=abc.ABCMeta):
    def __init__(self, config):
        self.extension_name = config['type']

    def set_mode(self, mode):
        self.mode = mode

    def is_train(self):
        return self.mode == 'train'
    def is_valid(self):
        return self.mode == 'valid'
    def is_test(self):
        return self.mode == 'test'

    #@abc.abstractmethod
    #def get_output_size(self):
    #    raise NotImplementedError()

    @abc.abstractmethod
    def __call__(self, stage_input):
        raise NotImplementedError()

def register_extension(type_):
    def _register_extension(klass):
        GeneralDataset.extensions[type_] = klass
        return klass
    return _register_extension

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