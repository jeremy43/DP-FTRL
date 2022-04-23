# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The DP-FTRL optimizer."""

import torch

__all__ = ['FTRLOptimizer']


class FTRLOptimizer(torch.optim.Optimizer):
    def __init__(self, params, momentum: float, record_last_noise: bool = True):
        """
        :param params: parameter groups
        :param momentum: if non-zero, use DP-FTRLM
        :param record_last_noise: whether to record the last noise. for the tree completion trick.
        """
        self.momentum = momentum
        self.record_last_noise = record_last_noise
        super(FTRLOptimizer, self).__init__(params, dict())

    def __setstate__(self, state):
        super(FTRLOptimizer, self).__setstate__(state)

    @torch.no_grad()
    def step(self, args, closure=None):
        alpha, noise = args
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p, nz in zip(group['params'], noise):
                if p.grad is None:
                    continue
                d_p = p.grad
                param_state = self.state[p]

                if len(param_state) == 0:
                    param_state['grad_sum'] = torch.zeros_like(d_p, memory_format=torch.preserve_format)
                    param_state['model_sum'] = p.detach().clone(memory_format=torch.preserve_format)  # just record the initial model
                    param_state['momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if self.record_last_noise:
                        param_state['last_noise'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # record the last noise needed, in order for restarting

                gs, ms = param_state['grad_sum'], param_state['model_sum']
                if self.momentum == 0:
                    gs.add_(d_p)
                    p.copy_(ms + (-gs - nz) / alpha)
                else:
                    gs.add_(d_p)
                    param_state['momentum'].mul_(self.momentum).add_(gs + nz)
                    p.copy_(ms - param_state['momentum'] / alpha)
                if self.record_last_noise:
                    param_state['last_noise'].copy_(nz)
        return loss

    @torch.no_grad()
    def restart(self, last_noise=None):
        """
        Restart the tree.
        :param last_noise: the last noise to be added. If none, use the last noise recorded.
        """
        assert last_noise is not None or self.record_last_noise
        for group in self.param_groups:
            if last_noise is None:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    if len(param_state) == 0:
                        continue
                    param_state['grad_sum'].add_(param_state['last_noise'])  # add the last piece of noise to the current gradient sum
            else:
                for p, nz in zip(group['params'], last_noise):
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    if len(param_state) == 0:
                        continue
                    param_state['grad_sum'].add_(nz)


class InitOptimizer(torch.optim.Optimizer):
    def __init__(self, params, shapes, device, n_batch, std):
        """
        :param params: parameter groups
        :param momentum: if non-zero, use DP-FTRLM
        :param record_last_noise: whether to record the last noise. for the tree completion trick.
        """
        self.shapes = shapes
        self.device = device
        self.std = std
        self.sgd_std = std * n_batch # std for sgd
        #self.mean_grad = [torch.zeros(shape).to(device) for shape in shapes]
        self.mean_grad = [torch.normal(0, self.std, shape).to(device) for shape in shapes]
        self.n_batch = n_batch
        super(InitOptimizer, self).__init__(params, dict())

    def __setstate__(self, state):
        super(InitOptimizer, self).__setstate__(state)

    @torch.no_grad()
    def step(self, args, closure=None):
        warmup, alpha = args
        # learning rate shall divide num_batches for sgd update
        alpha = alpha
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p, mean_g, shape in zip(group['params'], self.mean_grad, self.shapes):
                if p.grad is None:
                    continue
                # d_p = p.grad
                param_state = self.state[p]
                if len(param_state) == 0:
                    # Add the mean of gradient into the gradient sum
                    param_state['grad_sum'] = mean_g.clone(memory_format=torch.preserve_format)
                    param_state['model_sum'] = p.detach().clone(memory_format=torch.preserve_format)
                if warmup:
                    nz = torch.normal(0, self.sgd_std, shape).to(self.device)
                    p.add((-p.grad - nz) * alpha)
                gs = param_state['grad_sum']
                gs.add_(p.grad/(self.n_batch))

        return loss

class SAGOptimizer(torch.optim.Optimizer):
    def __init__(self, params, mean_grad, momentum: float, record_last_noise: bool = True):
        """
        :param params: parameter groups
        :param momentum: if non-zero, use DP-FTRLM
        :param record_last_noise: whether to record the last noise. for the tree completion trick.
        """
        self.momentum = momentum
        self.record_last_noise = record_last_noise
        self.mean_grad = mean_grad
        super(SAGOptimizer, self).__init__(params, dict())

    def __setstate__(self, state):
        super(SAGOptimizer, self).__setstate__(state)

    @torch.no_grad()
    def step(self, args, closure=None):
        alpha, noise, diff_gradient = args
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p, nz, diff_g, mean_g in zip(group['params'], noise, diff_gradient, self.mean_grad):
                if p.grad is None:
                    continue
                #d_p = p.grad
                param_state = self.state[p]

                if len(param_state) == 0:
                    print('yes, initialized with mean')
                    # Add the mean of gradient into the gradient sum
                    param_state['grad_sum'] = mean_g.clone(memory_format=torch.preserve_format)
                    #param_state['grad_sum'] = torch.zeros_like(diff_g, memory_format=torch.preserve_format)
                    param_state['model_sum'] = p.detach().clone(memory_format=torch.preserve_format)  # just record the initial model
                    param_state['momentum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if self.record_last_noise:
                        param_state['last_noise'] = torch.zeros_like(p, memory_format=torch.preserve_format)  # record the last noise needed, in order for restarting
                gs, ms = param_state['grad_sum'], param_state['model_sum']
                if self.momentum == 0:
                    gs.add_(diff_g)
                    p.add_((-gs - nz) * alpha)
                    #p.copy_(ms + (-gs - nz) * alpha)
                else:
                    raise NotImplementedError
                    #gs.add_(d_p)
                    param_state['momentum'].mul_(self.momentum).add_(gs + nz)
                    p.copy_(ms - param_state['momentum'] / alpha)
                if self.record_last_noise:
                    param_state['last_noise'].copy_(nz)
        return loss

    @torch.no_grad()
    def restart(self, last_noise=None):
        """
        Restart the tree.
        :param last_noise: the last noise to be added. If none, use the last noise recorded.
        """
        assert last_noise is not None or self.record_last_noise
        for group in self.param_groups:
            if last_noise is None:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    if len(param_state) == 0:
                        continue
                    param_state['grad_sum'].add_(param_state['last_noise'])  # add the last piece of noise to the current gradient sum
            else:
                for p, nz in zip(group['params'], last_noise):
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    if len(param_state) == 0:
                        continue
                    param_state['grad_sum'].add_(nz)
