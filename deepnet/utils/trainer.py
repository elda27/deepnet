
import chainer
from chainer import cuda, serializers
from chainer import computational_graph as cg
import numpy as np
import tqdm
import sys
import copy
from deepnet import utils
import os.path
import subprocess

class Trainer:
    def __init__(self, network, train_iter, valid_iter, visualizers, train_config, optimizer, logger, archive_dir, archive_nodes):
        self.network = network
        self.train_config = train_config
        
        self.n_max_train_iter = train_config['n_max_train_iter']
        self.n_max_valid_iter = train_config['n_max_valid_iter'] if train_config['n_max_valid_iter'] is not None else len(valid_iter.dataset)
        self.n_valid_step = train_config['n_valid_step']
        
        self.progress_vars = train_config['progress_vars']
        
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        
        self.archive_dir = archive_dir
        self.archive_nodes = archive_nodes

        self.visualizers = visualizers
        
        self.optimizer = optimizer
        
        self.logger = logger
        self.dump_variables = []
        for l in self.logger:
            for var_name in l.dump_variables:
                pos = var_name.find('.')
                if pos == -1:
                    self.dump_variables.append(var_name)
                else:
                    self.dump_variables.append(var_name[pos+1:])
        self.dump_variables = list(set(self.dump_variables))

    def train(self):
        with tqdm.tqdm(total=self.n_max_train_iter) as pbar:
            for i, batch in enumerate(self.train_iter):
                self.train_iteration = i
                variables = {}
                variables['__iteration__'] = i
                variables['__train_iteration__'] = self.train_iteration

                input_vars = self.batch_to_vars(batch)

                # Inference current batch.
                for stage_input in input_vars:
                    self.inference(stage_input, is_train=True)
                
                # Update variables.
                variables.update(self.network.variables)

                # Back propagation and update network 
                for loss_name, optimizer in self.optimizer.items():
                    if loss_name not in variables:
                        unreached = self.network.validate_network(loss_name)
                        raise ValueError(
                            'Unreached loss computation.\nFollowing list is not reached nodes: \n' + 
                            '\n'.join([ str(n) for n in  unreached ])
                            )

                    loss = variables[loss_name]
                    if i == 0:
                        self.write_network_architecture(os.path.join(self.archive_dir, 'model_{}.dot'.format(loss_name)), loss)
                    
                    self.network.update()
                    loss.backward()
                    optimizer.update()

                # Update variables and unwrapping chainer variable
                for var_name, value in variables.items():
                    variables[var_name] = utils.unwrapped(value)
                variables.update({ 'train.' + name: utils.unwrapped(value) for name, value in self.network.variables.items() })

                # validation if current iteraiton is multiplier as n_valid_step
                if i % self.n_valid_step == 0:
                    valid_variables = self.validate(variables=variables)
                    variables.update({ 'valid.' + name: value for name, value in valid_variables.items() })

                # Write log
                for logger in self.logger:
                    logger(variables, is_valid=False)

                # Update progress bar
                self.print_description(pbar, variables)
                pbar.update()
                if self.n_max_train_iter <= i:
                    break

                
    def validate(self, variables):
        valid_variables = dict()
        with tqdm.tqdm(total=self.n_max_valid_iter) as pbar, \
            chainer.no_backprop_mode():
            self.valid_iter.reset()
            for i, batch in enumerate(self.valid_iter):
                self.valid_iteration = i
                variables['__iteration__'] = i
                variables['__valid_iteration__'] = self.valid_iteration

                input_vars = self.batch_to_vars(batch)

                # Inference
                for j, stage_input in enumerate(input_vars):
                    self.inference(stage_input, is_train=False)
                    variables['__stage__'] = j
                    variables.update(self.network.variables)
                
                for visualizer in self.visualizers:
                    visualizer(variables)
                
                # Update variables
                for var_name in self.dump_variables:
                    var = variables[var_name]
                    if var_name not in valid_variables: # Initialize variable
                        if isinstance(var, chainer.Variable):
                            valid_variables[var_name] = chainer.functions.copy(var, -1)
                        else:
                            valid_variables[var_name] = var
                    else:
                        if isinstance(var, chainer.Variable):
                            valid_variables[var_name] += chainer.functions.copy(var, -1)
    
                pbar.update(self.valid_iter.batch_size)
                if self.n_max_valid_iter <= (i + 1) * self.valid_iter.batch_size:
                    break
            pbar.close()
        
        for node_name in self.archive_nodes:
            serializers.save_npz(
                os.path.join(self.archive_dir, node_name +'_{:08d}.npz'.format(variables['__train_iteration__'])), 
                self.network.network[node_name].model
                )

        # Compute mean variables
        for var_name in self.dump_variables:
            var = valid_variables[var_name]
            denom = float(self.n_max_valid_iter) / self.valid_iter.batch_size
            if isinstance(var, chainer.Variable):
                if self.train_config['gpu'] >= 0:
                    valid_variables[var_name] = float(chainer.cuda.to_cpu((var / denom).data))
                else:
                    valid_variables[var_name] = float((var / denom).data)
        # Save visualized results
        for visualizer in self.visualizers:
            visualizer.save()

        return valid_variables

    def print_description(self, pbar, variables):
        disp_vars = {}
        display_var_formats = []
        for var_format in self.progress_vars:
            # 
            var_name = ''
            pos = var_format.find(':')
            if pos == -1:
                var_name = var_format
            else:
                var_name = var_format[:pos]

            # cast variable
            var = variables[var_name]
            display_var_formats.append(var_name + '=' +'{' + var_format + '}')
            if isinstance(var, chainer.Variable):
                value = None
                if self.train_config['gpu'] >= 0:
                    value = chainer.cuda.to_cpu(var.data)
                else:
                    value = var.data
                if isinstance(value, np.ndarray):
                    if value.ndim == 0:
                        disp_vars[var_name] = float(value)
                    else:
                        disp_vars[var_name] = value.to_list()
            else:
                disp_vars[var_name] = var
        
        display_format = 'train[' + ','.join(display_var_formats) + ']'
        pbar.set_description(display_format.format(**disp_vars, __iteration__=variables['__iteration__']))

    def batch_to_vars(self, batch):
        # batch to vars
        input_vars = [ dict() for elem in batch[0] ]
        for elem in batch:              # loop about batch
            for i, stage_input in enumerate(elem):    # loop about stage input
                for name, input_ in stage_input.items():
                    input_vars[i].setdefault(name, []).append(input_)
        return input_vars

    def inference(self, stage_input, is_train=False):
        self.network(mode='train' if is_train else 'valid', **stage_input)

    def write_network_architecture(self, graph_filename, loss):
        with open(graph_filename, 'w+') as o:
            o.write(cg.build_computational_graph((loss, )).dump())

        try:
            subprocess.call('dot -T png {} -o {}'.format(graph_filename, 
                            graph_filename.replace('.dot', '.png')), 
                            shell=True)
        except:
            warnings.warn('please install graphviz and set your environment.')