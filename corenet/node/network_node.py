
class NetworkNode:
    DeclareArguments = {}

    def __init__(
        self, name, model,
        input, output,
        args, **kwargs
    ):
        input_ = input
        self.name  = name

        self.input = input_ if isinstance(input_, list) else [ input_ ] 
        self.output = output if isinstance(output, list) else [ output ] 

        def recusrsive_check_instance(a):
            if isinstance(a, (list, tuple)):
                return all([ recusrsive_check_instance(e) for e in a ])
            else:
                return isinstance(e, str)

        assert recusrsive_check_instance(self.input),  'Input must be string: {}'.format(self.input)
        assert recusrsive_check_instance(self.output), 'Output must be string: {}'.format(self.output)

        self.model = model
        self.args = args
        self.attrs = kwargs
    
    def __call__(self, *args):
        return self.model(*args, **self.attrs)

    def __repr__(self):
        return str(dict(input=self.input, output=self.output))

    def get_internal_param(self, fields):
        def get_field(var, fileds):
            if len(fields) == 0:
                return var
            return get_field(getattr(var, fields[0]), fields[1:])

        proc = get_field(self.model, fields[:-1])
        if not hasattr(proc, 'stores'):
            raise NotImplementedError('get_internal_param method is not fully implementation')
        return proc.stores[fields[-1]]

    def get_attr(self, key):
        return self.attrs[key]
