from abc import abstractmethod
import os.path
from contextlib import ExitStack
import chainer
from itertools import zip_longest

class Logger:
    def __init__(self, output_filename, dump_variables, variable_weights):
        self.output_filename = output_filename
        self.dump_variables = dump_variables
        
        if os.path.exists(output_filename):
            os.remove(output_filename)

    def __call__(self, variables, is_valid=False):
        dump_vars = {}
        for var_name, weight in zip_longest(self.dump_variables, self.weights):
            if var_name not in variables:
                data[var_name] = ''
            
            if weight is not None or (isinstance(weight, str) and weight != ''):
                data[var_name] = variables[var_name] * float(weight)
                data[var_name + '.raw'] = variables[var_name]
            else:
                data[var_name] = variables[var_name]

        dump_vars = { var_name:variables[var_name] if var_name in variables else '' for var_name in self.dump_variables }
        self.dump(**dump_vars)

    @abstractmethod
    def dump(self, **kwargs):
        raise NotImplementedError()

class CsvLogger(Logger):
    def dump(self, **kwargs):
        with ExitStack() as estack:
            fp = None
            if not os.path.exists(self.output_filename):
                fp = estack.enter_context(open(self.output_filename, 'w+'))
                fp.write(','.join(self.dump_variables) + '\n')
            else:
                fp = estack.enter_context(open(self.output_filename, 'a'))

            fp.write(','.join([str(kwargs[var_name]) for var_name in self.dump_variables]) + '\n')
