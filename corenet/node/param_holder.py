from abc import abstractmethod


class ParamHolder:
    def __init__(self, *internal_params):
        self.internal_params = internal_params
        self.hold_params = {}

    def request_param(self, param_name):
        if param_name not in self.internal_params:
            raise KeyError('Unknown param name.')
        self.hold_params[param_name] = None

    def add_param(self, name, param):
        if name not in self.hold_params:
            return
        self.hold_params[name] = param

    def clear_holds(self):
        self.hold_params = {}

    def get_param(self, name):
        return self.hold_params[name]

