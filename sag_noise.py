
"""The tree aggregation protocol for noise addition in DP-FTRL."""

import torch
from collections import namedtuple

from absl import app


class TableTorch:
    @torch.no_grad()
    def __init__(self, n, shapes, device, n_batch, test_mode=False):
        """
        This table records the latest gradient of n data
        :param n: number of data point
        :param shapes: shapes of the gradient, which is basically shape of the gradients
        :param device: device for pytorch tensor
        :param test_mode: if in test mode, noise will be 1 in each node of the tree
        """
        assert n > 0
        self.n = n
        self.shapes = shapes
        self.device = device
        self.n_batch = n_batch
        # step is the pointer to the next gradient need to change.
        self.step = 0
        # shall we detach it? record n*d
        self.recorded = []
        #self.recorded = [[torch.zeros(shape).to(self.device) for shape in shapes]]
        self.test_mode = test_mode

    @torch.no_grad()
    def __call__(self, new_param, clip_gradient=None, init=False, ret_old = True):
        """
        :return: the i-th gradient in the last iteration and update it.
        """

        idx = self.step % self.n_batch
        if init is True:
            # Add new gradient into the list
            self.recorded.append([torch.zeros(shape).to(self.device) for shape in self.shapes])
            for p, old_g, new_p in zip(new_param, self.recorded[idx], clip_gradient):
                if p.grad is None:
                    continue

                old_g.copy_(new_p) # new_p.grad already divdes n_batch
            self.step += 1
            return self.recorded[idx]
        else:
            if ret_old is True:
                ret_copy = [old_g.clone() for old_g in self.recorded[idx]]
                # return the old copy
                """
                ret_copy = [torch.zeros(shape).to(self.device) for shape in self.shapes]
                for new_p, old_g, ret_g in zip(new_param, self.recorded[idx], ret_copy):
                    if new_p.grad is None:
                        continue
                    new_g = new_p.grad *1.0/self.n_batch
                    diff = (new_g - old_g)
                    ret_g.copy_()
                    old_g.copy_(new_g)
                """
                return ret_copy
            else:
                # Receive a new copy of gradient
                for new_p, old_g, ret_g in zip(new_param, self.recorded[idx], clip_gradient):
                    if new_p.grad is None:
                        continue
                    old_g.copy_(ret_g)


                self.step += 1
            #return ret_copy

"""

class TableTorch:
    @torch.no_grad()
    def __init__(self, n, shapes, device, n_batch, test_mode=False):
       
        This table records the latest gradient of n data
        :param n: number of data point
        :param shapes: shapes of the gradient, which is basically shape of the gradients
        :param device: device for pytorch tensor
        :param test_mode: if in test mode, noise will be 1 in each node of the tree
      
        assert n > 0
        self.n = n
        self.shapes = shapes
        self.device = device
        self.n_batch = n_batch
        # step is the pointer to the next gradient need to change.
        self.step = 0
        # shall we detach it? record n*d
        self.recorded = []
        #self.recorded = [[torch.zeros(shape).to(self.device) for shape in shapes]]
        self.test_mode = test_mode

    @torch.no_grad()
    def __call__(self, new_param, init=False):
        
        :return: the i-th gradient in the last iteration and update it.
   

        idx = self.step % self.n_batch
        if idx == 0:
            print('now idx is 0')
        if init is True:
            # Add new gradient into the list
            self.recorded.append([torch.zeros(shape).to(self.device) for shape in self.shapes])
            for new_p, old_g in zip(new_param, self.recorded[idx]):
                if new_p.grad is None:
                    continue
                old_g.copy_(new_p.grad/self.n_batch) # new_p.grad already divdes n_batch
            self.step += 1
            return self.recorded[idx]
        else:
            # return the old copy
            ret_copy = [torch.zeros(shape).to(self.device) for shape in self.shapes]
            for new_p, old_g, ret_g in zip(new_param, self.recorded[idx], ret_copy):
                if new_p.grad is None:
                    continue
                new_g = new_p.grad *1.0/self.n_batch
                diff = (new_g - old_g)
                ret_g.copy_(diff)
                old_g.copy_(new_g)

            self.step += 1
            return ret_copy
"""



class CummuNoiseTorch:
    @torch.no_grad()
    def __init__(self, std, shapes, device, test_mode=False):
        """
        :param std: standard deviation of the noise
        :param shapes: shapes of the noise, which is basically shape of the gradients
        :param device: device for pytorch tensor
        :param test_mode: if in test mode, noise will be 1 in each node of the tree
        """
        assert std >= 0
        self.std = std
        self.shapes = shapes
        self.device = device
        self.step = 0
        self.binary = [0]
        self.noise_sum = [torch.zeros(shape).to(self.device) for shape in shapes]
        self.recorded = [[torch.zeros(shape).to(self.device) for shape in shapes]]
        self.test_mode = test_mode

    @torch.no_grad()
    def __call__(self):
        """
        :return: the noise to be added by DP-FTRL
        """
        if self.std <= 0 and not self.test_mode:
            return self.noise_sum

        self.step += 1
        # self.record: record noise
        idx = 0
        # Originally, we need to instore 2^bit noise. The following algorithm only needs to store bit noise. this is
        # because only the latest left node noise is instored.
        # Question: what's idx? I suppose idx be the pointer in binary number.
        # suppose last round we calculate the 7-th gradient, whose binary number is 111
        # when we compute the next one 1000, we need to flag the last three digits and remove the corresponding noise.
        while idx < len(self.binary) and self.binary[idx] == 1:
            self.binary[idx] = 0
            for ns, re in zip(self.noise_sum, self.recorded[idx]):
                ns -= re
            idx += 1
        if idx >= len(self.binary):
            self.binary.append(0)
            self.recorded.append([torch.zeros(shape).to(self.device) for shape in self.shapes])

        for shape, ns, re in zip(self.shapes, self.noise_sum, self.recorded[idx]):
            if not self.test_mode:
                n = torch.normal(0, self.std, shape).to(self.device)
            else:
                n = torch.ones(shape).to(self.device)
            ns += n
            re.copy_(n)

        self.binary[idx] = 1
        return self.noise_sum

    @torch.no_grad()
    def proceed_until(self, step_target):
        """
        Proceed until the step_target-th step. This is for the binary tree completion trick.

        :return: the noise to be added by DP-FTRL
        """
        if self.step >= step_target:
            raise ValueError(f'Already reached {step_target}.')
        while self.step < step_target:
            noise_sum = self.__call__()
        return noise_sum


Element = namedtuple('Element', 'height value')


class CummuNoiseEffTorch:
    """
    The tree aggregation protocol with the trick in Honaker, "Efficient Use of Differentially Private Binary Trees", 2015
    """
    @torch.no_grad()
    def __init__(self, std, shapes, device):
        """
        :param std: standard deviation of the noise
        :param shapes: shapes of the noise, which is basically shape of the gradients
        :param device: device for pytorch tensor
        """
        self.std = std
        self.shapes = shapes
        self.device = device

        self.step = 0
        self.noise_sum = [torch.zeros(shape).to(self.device) for shape in shapes]
        self.stack = []

    @torch.no_grad()
    def get_noise(self):
        return [torch.normal(0, self.std, shape).to(self.device) for shape in self.shapes]

    @torch.no_grad()
    def push(self, elem):
        for i in range(len(self.shapes)):
            self.noise_sum[i] += elem.value[i] / (2.0 - 1 / 2 ** elem.height)
        self.stack.append(elem)

    @torch.no_grad()
    def pop(self):
        elem = self.stack.pop()
        for i in range(len(self.shapes)):
            self.noise_sum[i] -= elem.value[i] / (2.0 - 1 / 2 ** elem.height)

    @torch.no_grad()
    def __call__(self):
        """
        :return: the noise to be added by DP-FTRL
        """
        self.step += 1

        # add new element to the stack
        self.push(Element(0, self.get_noise()))

        # pop the stack
        while len(self.stack) >= 2 and self.stack[-1].height == self.stack[-2].height:
            # create new element
            left_value, right_value = self.stack[-2].value, self.stack[-1].value
            new_noise = self.get_noise()
            new_elem = Element(
                self.stack[-1].height + 1,
                [x + (y + z) / 2 for x, y, z in zip(new_noise, left_value, right_value)])

            # pop the stack, update sum
            self.pop()
            self.pop()

            # append to the stack, update sum
            self.push(new_elem)
        return self.noise_sum

    @torch.no_grad()
    def proceed_until(self, step_target):
        """
        Proceed until the step_target-th step. This is for the binary tree completion trick.

        :return: the noise to be added by DP-FTRL
        """
        if self.step >= step_target:
            raise ValueError(f'Already reached {step_target}.')
        while self.step < step_target:
            noise_sum = self.__call__()
        return noise_sum


def main(argv):
    # This is a small test. If we set the noise in each node as 1 (by setting
    # test_mode=True), we should be seeing the returned noise as the number of
    # 1s in the binary representations of i when cummu_noises is called i times.

    def countSetBits(n):
        count = 0
        while (n):
            n &= (n - 1)
            count += 1
        return count
    cummu_noises = CummuNoiseTorch(1.0, [(1,)], 'cuda', test_mode=True)
    for epoch in range(31):
        random_noise = cummu_noises()
        assert random_noise[0].cpu().numpy()[0] == countSetBits(epoch + 1)


if __name__ == '__main__':
    app.run(main)
