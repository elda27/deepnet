import abc
import enum

class PostProcessTrigger(enum.IntEnum):
    AfterEachProcess = 1 # Post process after each process.
    AfterProcess     = 2 # Post process after sequence (e.g. after validation).
    Both             = 3 # Post process after each process and sequence.

class PostProcessor:
    def __init__(self, process_trigger, store_trigger, training = False, validation = True, test = True):
        self.process_trigger = process_trigger
        self.store_trigger = store_trigger
        self.training = training
        self.validation = validation
        self.test = test

    def peek(self, variable):
        raise NotImplementedError()

    def process(self, variable):
        raise NotImplementedError()

    def update(self, variable, mode, in_processing = True):
        if mode == 'train' and not self.training:
            return
        if mode == 'valid' and not self.validation:
            return
        if mode == 'test' and not self.test:
            return

        ppt = PostProcessTrigger
        if in_processing and self.process_trigger in (ppt.AfterEachProcess, ppt.Both):
            self.peek(variable)

        elif not in_processing and self.process_trigger in (ppt.AfterProcess, ppt.Both) :
            self.process(variable)

        return self.store_trigger in (ppt.AfterEachProcess, ppt.Both) if in_processing else self.store_trigger in (ppt.AfterProcess, ppt.Both)


    def get_result(self):
        raise NotImplementedError()

_registered_postprocess = {}

def register_postprocess(name):
    def _register_postprocess(klass):
        _registered_postprocess[name] = klass
        return klass
    return _register_postprocess

def create_postprocess(name, *args, **kwargs):
    return _registered_postprocess[name](*args, **kwargs)

class PostProcessManager:
    def __init__(self, config_fields):
        self.postprocess_list = []
        for config in config_fields:
            self.postprocess_list.append(
                create_postprocess(config.pop('type'), **config)
            )

    def __call__(self, variable, mode, peek=True):
        for post_process in self.postprocess_list:
            if post_process.update(variable, mode, peek):
                variable.update(post_process.get_result())
    