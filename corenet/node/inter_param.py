

def __get_internal_param(self, param_name):
    return self._stores[param_name]

def __set_internal_param(self, param_name, value):
    if param_name in self._requests:
        self._stores[param_name] = value

def __request_internal_param(self, param_name):
    return self._requests.append(param_name)

def edit_function(klass, method_name, method):
    if isinstance(method, str):
        if not hasattr(klass, method):
            setattr(klass, method_name, method)
    return getattr(klass, method_name)

def declare_internal_param(
    *param_name, 
    getter='__get_internal_param',
    setter='__set_internal_param',
    requester='__request_internal_param',
):
    def _declare_internal_param(klass):
        getter = edit_function(klass, method_name, __get_internal_param)
        setter = edit_function(klass, method_name, __get_internal_param)
        requester = edit_function(klass, method_name, __get_internal_param)