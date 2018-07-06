import toml
import re
import os.path

def load(filename, **kwargs):
    with open(filename, 'r') as fp:
        return loads(fp.read(), root_dir=os.path.dirname(filename), **kwargs)

def loads(s, **kwargs):
    parser = ConfigParser(s, **kwargs)
    return parser.config

def expand_variable(config):
    parser = ConfigParser(None)
    parser.config = config
    parser.expand()
    return parser.config

def find_include_file(filename, root_dir = None):
    if root_dir is not None and os.path.exists(os.path.join(root_dir, filename)):
        return os.path.join(root_dir, filename)
    elif os.path.exists(filename):
        return filename
    else:
        raise AttributeError('Including file is not found: ' + filename)


def expand_include(config, root_dir = None):
    if 'include' not in config['config']:
        return config
    includes = config['config']['include']
    if not isinstance(includes, list):
        includes = [ includes ]

    included_config = {}

    load_configs = [ ConfigParser(find_include_file(include, root_dir), is_expand_variable=False).config for include in includes ]
    
    load_configs.append(config)
    for including_config in load_configs:
        for key in including_config:
            if isinstance(including_config[key], list):
                included_config.setdefault(key, []).extend(including_config[key])
            elif isinstance(including_config[key], dict):
                included_config.setdefault(key, {}).extend(including_config[key])
            else:
                included_config[key] = including_config[key]

    return included_config

class ConfigParser:
    def __init__(self, config_string, is_variable_expansion = True, root_dir=None):
        self.variable_pattern = re.compile(r'\$\{?(?P<variable>[\w\d_\.]+)\}?')
        if config_string is None:
            self.config = None
            return

        self.config = toml.loads(config_string)
        
        self.config = expand_include(self.config, root_dir)
        if is_variable_expansion:
            self.expand()
        

    def expand(self):
        for key, value in self.config.items():
            self.config[key] = self.recursive_apply_variables(value)

    def try_cast(self, value, src_value):
        if isinstance(src_value, (float, int)):
            try:
                t = type(src_value)
                return t(value)
            except:
                pass

        return value

    def recursive_apply_variables(self, field):
        if isinstance(field, dict):
            for key, value in field.items():
                field[key] = self.recursive_apply_variables(value)
        elif isinstance(field, list):
            for i in range(len(field)):
                field[i] = self.recursive_apply_variables(field[i])
        else:
            field = self.apply_variables(field)
        return field

    def apply_variables(self, value):
        if not isinstance(value, str):
            return value
        matches = re.findall(self.variable_pattern, value)
        for match in matches:
            value = self.replace_variable(value, match)
        return value

    def get_value(self, dict_, var_key):
        keys = var_key.split('.')
        for key in keys:
            if key not in dict_:
                raise toml.TomlDecodeError('Unknown variable: {}, '.format(var_key))
            dict_ = dict_[key]
        return dict_

    def replace_variable(self, value, var_name):
        var_value = self.get_value(self.config, var_name)
        assert not isinstance(var_value, dict), 'Currently, not supported dict variable expansion.'
        if isinstance(var_value, list):
            result = []
            for v in var_value:
                assert not isinstance(v, list)
                replaced_v = value.replace('${' + var_name + '}', str(v))
                result.append(self.try_cast(replaced_v, v))
            return result
        else:
            replaced_value = value.replace('${' + var_name + '}', str(var_value))
            return self.try_cast(replaced_value, var_value)
                
    def check_variable(self, string):
        if not isinstance(string, str):
            return None

        matched = re.findall(self.variable_pattern, string)
        if len(matched) == 0:
            return None

        return matched
