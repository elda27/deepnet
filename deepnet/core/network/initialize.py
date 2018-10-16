from deepnet.core.registration import register_initialize_field, _registered_initialize_field
from deepnet.core.config import get_global_config
from deepnet.core.network.build import get_process
from deepnet.utils import get_field

from chainer.serializers import load_npz
import os.path
import glob


def initialize_networks_(log_root_dir, step_index, config):
    """Initialize network

    Args:
        log_root_dir (str): Root directory of the log.
        stage_index (int): Index of the learning step.
        config (dict): Configuration of the network.
    """

    if 'initialize' not in config:
        return

    initialize_fields = config['initialize']
    for field in initialize_fields:
        _registered_initialize_field[field['mode']](field)


def initialize_networks(**kwargs):
    _registered_initialize_field[kwargs.pop('mode')](**kwargs)


@register_initialize_field('load')
def initialize_prelearned_model(**field):
    log_root_dir = get_global_config('log.root')
    step_index = get_global_config('step_index')
    if 'from_step' in field:
        step_index = field['from_step']

    name = field['name']
    created_model = get_process(name)

    model_glob_str = os.path.join(
        log_root_dir, 'model_step' + str(step_index), name + '_*.npz')
    found_models = glob.glob(model_glob_str)
    assert len(found_models) != 0, 'Model not found:' + model_glob_str
    archive_filename = found_models[-1]

    load_npz(archive_filename, created_model)


@register_initialize_field('share')
def shared_layer(**field):
    #get_field = deepnet.utils.get_field
    to_fields = field['to']
    from_fields = field['from']
    freeze_layer = field.get('freeze', False)

    if not isinstance(to_fields, list):
        to_fields = [to_fields]

    if not isinstance(from_fields, list):
        from_fields = [from_fields]

    for to_field, from_field in zip():
        to_names = to_field.split('.')
        from_names = from_field.split('.')

        from_model = get_process(from_names[0])
        to_model = get_process(to_names[0])

        from_layer = get_field(from_model, from_names[1:])
        to_parent_layer = get_field(to_model, to_names[1:-1])

        with to_model.init_scope():
            if field.get('deepcopy', False):
                if hasattr(to_parent_layer, to_names[-1]):
                    raise AttributeError(
                        'Destination layer doesn\'t have attribute: {}'.format(to_names[-1]))
                to_layer = getattr(to_parent_layer, to_names[-1])
                from_layer.copyparams(to_layer)

                if hasattr(to_parent_layer, 'layers'):
                    to_parent_layer.layers[to_names[-1]] = to_layer
            else:
                setattr(to_parent_layer, to_names[-1], from_layer)
                if hasattr(to_parent_layer, 'layers'):
                    to_parent_layer.layers[to_names[-1]] = from_layer
