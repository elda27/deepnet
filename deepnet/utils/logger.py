from abc import abstractmethod
import os.path
from contextlib import ExitStack
import chainer
from itertools import zip_longest
from collections import OrderedDict

class Logger:
    def __init__(self, output_filename, dump_variables, variable_weights):
        self.output_filename = output_filename
        self.dump_variables = dump_variables
        self.weights = variable_weights
        
        if os.path.exists(output_filename):
            os.remove(output_filename)

    def __call__(self, variables, is_valid=False):
        dump_vars = OrderedDict()
        added_vars = OrderedDict()

        for var_name, weight in zip_longest(self.dump_variables, self.weights):
            if var_name not in variables:
                dump_vars[var_name] = ''
                if weight is not None and not (isinstance(weight, str) and weight == ''):
                    added_vars[var_name + '.raw'] = ''
                continue
            
            if weight is not None and not (isinstance(weight, str) and weight == ''):
                dump_vars[var_name] = variables[var_name] * float(weight)
                added_vars[var_name + '.raw'] = variables[var_name]
            else:
                dump_vars[var_name] = variables[var_name]
        
        dump_vars.update(added_vars)
        self.dump(dump_vars)

    @abstractmethod
    def dump(self, dump_vars):
        raise NotImplementedError()

class CsvLogger(Logger):
    def dump(self, dump_vars):
        with ExitStack() as estack:
            fp = None
            if not os.path.exists(self.output_filename):
                fp = estack.enter_context(open(self.output_filename, 'w+'))
                fp.write(','.join(dump_vars) + '\n')
            else:
                fp = estack.enter_context(open(self.output_filename, 'a'))

            fp.write(','.join([ str(dump_vars[var_name]) for var_name in dump_vars ]) + '\n')
