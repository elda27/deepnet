from abc import abstractmethod
import os.path
from contextlib import ExitStack
import chainer

class Logger:
    def __init__(self, output_filename, dump_variables):
        self.output_filename = output_filename
        self.dump_variables = dump_variables
        if os.path.exists(output_filename):
            os.remove(output_filename)

    def __call__(self, variables, is_valid=False):
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
