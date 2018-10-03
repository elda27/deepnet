from abc import abstractmethod
import os.path
from contextlib import ExitStack
import chainer
from itertools import zip_longest
from collections import OrderedDict
from deepnet.utils.functions import get_field
#from deepnet.core.registration import get_registered_process
#from deepnet.core.network.build import get_process
import deepnet.core.network.build
from deepnet import utils


class Logger:
    def __init__(self, output_filename, dump_variables, variable_weights):
        self.output_filename = output_filename
        self.dump_variables = dump_variables
        self.weights = variable_weights

        if os.path.exists(output_filename):
            os.remove(output_filename)

    def __call__(self, variables, is_valid=False):
        dump_vars = OrderedDict()
        appendix_vars = OrderedDict()

        for var_name, weight in zip_longest(self.dump_variables, self.weights):
            if var_name not in variables:
                dump_vars[var_name] = ''
                appendix_vars['raw.{}'.format(var_name)] = ''
                appendix_vars['weight.{}'.format(var_name)] = ''
                continue

            if isinstance(weight, str):
                if weight == '':
                    weight = None
                elif weight in variables:
                    member = weight.split('.')
                    var_name = member[0]
                    fields = member[1:]
                    weight = get_field(variables[var_name], fields)
                else:
                    member = weight.split('.')
                    name = member[0]
                    fields = member[1:]
                    try:
                        process = deepnet.core.network.build.get_process(name)
                        weight = get_field(process, fields)
                        weight = utils.unwrapped(weight)
                    except:
                        weight = None

            if weight is not None:
                dump_vars[var_name] = variables[var_name] * float(weight)
                appendix_vars['raw.{}'.format(var_name)] = variables[var_name]
                appendix_vars['weight.{}'.format(var_name)] = float(weight)
            else:
                dump_vars[var_name] = variables[var_name]

        dump_vars.update(appendix_vars)
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

            fp.write(','.join([str(dump_vars[var_name])
                               for var_name in dump_vars]) + '\n')
