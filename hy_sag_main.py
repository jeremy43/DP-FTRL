"""DP-SAG training, based on paper
"Practical and Private (Deep) Learning without Sampling or Shuffling"
https://arxiv.org/abs/2103.00039.
This is for hyper-parameter tuning
"""


from absl import app
from absl import flags

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import trange
import numpy as np
import copy
import tensorflow as tf
import torch
from torch.utils.tensorboard import SummaryWriter
from opacus import PrivacyEngine

from optimizers import FTRLOptimizer, SAGOptimizer, InitOptimizer
from sag_noise import CummuNoiseTorch, CummuNoiseEffTorch, TableTorch
from nn import get_nn
from data import get_data
import utils
from utils import EasyDict


FLAGS = flags.FLAGS

flags.DEFINE_enum('data', 'mnist', ['mnist', 'cifar10', 'emnist_merge'], '')

flags.DEFINE_boolean('dp_sag', True, 'If True, train with DP-sag. If False, train with vanilla SAG.')
flags.DEFINE_float('noise_multiplier', 1.0, 'Ratio of the standard deviation to the clipping norm.')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm.')

flags.DEFINE_integer('restart', 0, 'If > 0, restart the tree every this number of epoch(s).')
flags.DEFINE_boolean('effi_noise', False, 'If True, use tree aggregation proposed in https://privacytools.seas.harvard.edu/files/privacytools/files/honaker.pdf.')
flags.DEFINE_boolean('tree_completion', False, 'If true, generate until reaching a power of 2.')

flags.DEFINE_float('momentum', 0, 'Momentum for DP-FTRL.')
flags.DEFINE_float('learning_rate', 0.5, 'Learning rate.')
flags.DEFINE_integer('batch_size', 25, 'Batch size.')
flags.DEFINE_integer('epochs', 10, 'Number of epochs.')
flags.DEFINE_boolean('warmup',False, 'If True, use one path DPSGD before runing DP-SAG')
flags.DEFINE_integer('report_nimg', -1, 'Write to tb every this number of samples. If -1, write every epoch.')

flags.DEFINE_integer('run', 20, '(run-1) will be used for random seed.')
flags.DEFINE_string('dir', '.', 'Directory to write the results.')

# the algorithm guarantees (alpha, epoch*alpha*2*log(num_batch)/noise_multiplier**2)-RDP, and we assume runing
# for 5 epoches.




def main(argv):

    # Hyperparameters for training.

    batch = FLAGS.batch_size

    noise_multiplier = FLAGS.noise_multiplier if FLAGS.dp_sag else -1
    clip = FLAGS.l2_norm_clip if FLAGS.dp_sag else -1
    scale = [0.5**i for i in [ -1.,-0.5, 0, 0.5, 1]]
    if not FLAGS.restart:
        FLAGS.tree_completion = False
    for lr in scale:
        print('current learning rate', lr)
        train(lr,batch, noise_multiplier, clip)




def train(lr,  batch, noise_multiplier, clip):
    epochs = FLAGS.epochs
    tf.get_logger().setLevel('ERROR')
    tf.config.experimental.set_visible_devices([], "GPU")

    # Setup random seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(FLAGS.run - 1)
    np.random.seed(FLAGS.run - 1)

    # Data
    trainset, testset, ntrain, nclass = get_data(FLAGS.data)
    print('Training set size', trainset.image.shape)
    num_batches = ntrain // batch
    report_nimg = ntrain if FLAGS.report_nimg == -1 else FLAGS.report_nimg
    assert report_nimg % batch == 0

    # Get the name of the output directory.
    log_dir = os.path.join(FLAGS.dir, FLAGS.data,'correct_sag',
                           utils.get_fn(EasyDict(batch=batch),
                                        EasyDict(dpsgd=FLAGS.dp_sag, restart=FLAGS.restart, completion=FLAGS.tree_completion, noise=noise_multiplier, clip=clip, mb=1),
                                        [EasyDict({'lr': lr}),
                                         EasyDict(m=FLAGS.momentum if FLAGS.momentum > 0 else None,
                                                  effi=FLAGS.effi_noise),
                                         EasyDict(sd=FLAGS.run)]
                                        )
                           )
    print('Model dir', log_dir)

    # Class to output batches of data
    class DataStream:
        def __init__(self):
            self.shuffle()

        def shuffle(self):
            self.perm = np.random.permutation(ntrain)
            self.i = 0

        def __call__(self):
            if self.i == num_batches:
                self.i = 0
            batch_idx = self.perm[self.i * batch:(self.i + 1) * batch]
            self.i += 1
            return trainset.image[batch_idx], trainset.label[batch_idx]

    data_stream = DataStream()
    # Initialize the prev_grad table and return the mean of gradient.
    def train_init(model, optimizer, device, prev_grad, writer, shapes):
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        losses = []
        epoch = 0
        loop = trange(0, num_batches * batch, batch,
                      leave=False, unit='img', unit_scale=batch,
                      desc='Epoch %d/%d' % (1 + epoch, epochs))
        step = 0
        # Initialization: a full pass of gradient over all data points.
        model.train()
        print('nb_batch', len(loop))
        #mean_grad = [torch.zeros(shape).to(device) for shape in shapes]
        for it in loop:
            step += 1
            data, target = data_stream()
            data = torch.Tensor(data).to(device)
            target = torch.LongTensor(target).to(device)
            # compute gradient
            optimizer.zero_grad()
            output = model(data).to(device)
            loss = criterion(output, target)
            loss.backward()
            # if warmup is true, run one round DP-GD (without sampling)
            optimizer.step((FLAGS.warmup, lr))
            # Update table
            diff_gradient = prev_grad(model.parameters(), init=True)

            """
            for param1, param2 in zip(model.parameters(), mean_grad):
                if param1.grad is None:
                    continue
                new_grad = param1.grad.detach().clone(memory_format=torch.preserve_format)
                param2 += 1.0 / len(loop) * new_grad
            """
        # Compute the mean of gradient over n points
        acc_train, acc_test = test(model, device)
        writer.add_scalar('eval/accuracy_test', 100 * acc_test, step)
        writer.add_scalar('eval/accuracy_train', 100 * acc_train, step)
        print('Step %04d Accuracy %.2f' % (step, 100 * acc_test))
        writer.add_scalar('eval/loss_train', np.mean(losses), epoch + 1)
        print('Epoch %04d Loss %.2f' % (epoch + 1, np.mean(losses)))
        return optimizer.mean_grad, model

    # Function to conduct training for one epoch
    def train_loop(model,  device, prev_grad, optimizer, cumm_noise, epoch, writer):
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        losses = []
        loop = trange(0, num_batches * batch, batch,
                      leave=False, unit='img', unit_scale=batch,
                      desc='Epoch %d/%d' % (1 + epoch, epochs))
        step = epoch * num_batches
        for it in loop:
            step += 1
            data, target = data_stream()
            data = torch.Tensor(data).to(device)
            target = torch.LongTensor(target).to(device)

            optimizer.zero_grad()
            output = model(data).to(device)
            loss = criterion(output, target)
            loss.backward()
            diff_gradient = prev_grad(model.parameters(), init=False)

            optimizer.step((lr, cumm_noise(), diff_gradient))
            losses.append(loss.item())

            if (step * batch) % report_nimg == 0:
                acc_train, acc_test = test(model, device)
                writer.add_scalar('eval/accuracy_test', 100 * acc_test, step)
                writer.add_scalar('eval/accuracy_train', 100 * acc_train, step)
                model.train()
                print('Step %04d Accuracy %.2f' % (step, 100 * acc_test))

        writer.add_scalar('eval/loss_train', np.mean(losses), epoch + 1)
        print('Epoch %04d Loss %.2f' % (epoch + 1, np.mean(losses)))

    # Function for evaluating the model to get training and test accuracies
    def test(model, device, desc='Evaluating'):
        model.eval()
        b = 1000
        with torch.no_grad():
            accs = [0, 0]
            for i, dataset in enumerate([trainset, testset]):
                for it in trange(0, dataset.image.shape[0], b, leave=False, desc=desc):
                    data, target = dataset.image[it: it + b], dataset.label[it: it + b]
                    data, target = torch.Tensor(data).to(device), torch.LongTensor(target).to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    accs[i] += pred.eq(target.view_as(pred)).sum().item()
                accs[i] /= dataset.image.shape[0]
        return accs

    # Get model for different dataset
    #device = torch.device('cuda')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_nn({'mnist': 'small_nn',
                    'emnist_merge': 'small_nn',
                    'cifar10': 'vgg128'}[FLAGS.data],
                   nclass=nclass).to(device)
    model_copy = copy.deepcopy(model)
    # Initialize the mean_gradient in DP-SAG
    # (1) use one pass gradient to compute the mean of gradient and initialize the gradient_table
    # (2) Restore the mean of gradient in optimizer
    # (3) At iteration t, compute the gradient of batch i and add (g(i)^t - g(i)^{t-1})/num_batch
    # Use the CummuNoise module to generate the noise using the tree aggregation. The noise will be passed into the
    # optimizer. the noise scale shall divide by num_batch
    writer = SummaryWriter(os.path.join(log_dir, 'tb'))
    shapes = [p.shape for p in model.parameters()]
    prev_grad = TableTorch(noise_multiplier * clip / batch, shapes, device, num_batches)

    #mean_grad = train_init(model, device, prev_grad, writer, shapes)
    #mean_grad = [torch.zeros(shape).to(device) for shape in shapes]

    ini_optimizer = InitOptimizer(model_copy.parameters(), shapes, device, num_batches,noise_multiplier * clip /
                                  (batch*num_batches))
    privacy_engine = PrivacyEngine(model_copy, batch_size=batch, sample_size=ntrain, alphas=[], noise_multiplier=0,
                                   max_grad_norm=clip)
    privacy_engine.attach(ini_optimizer)
    """
    model_parameters = filter(lambda p: p.grad_sample, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('before num of params', params)
    """
    # without model_copy, there is some issue with opaque.
    mean_grad, init_model = train_init(model_copy, ini_optimizer, device, prev_grad, writer, shapes)

    model.load_state_dict(init_model.state_dict())
    #model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    #params = sum([np.prod(p.size()) for p in model_parameters])
    #print(' after num of params', params)
    privacy_engine.detach()

    optimizer = SAGOptimizer(model.parameters(), mean_grad, momentum=FLAGS.momentum,
                              record_last_noise=FLAGS.restart > 0 and FLAGS.tree_completion)
    if FLAGS.dp_sag:
        privacy_engine = PrivacyEngine(model, batch_size=batch, sample_size=ntrain, alphas=[], noise_multiplier=0, max_grad_norm=clip)
        privacy_engine.attach(optimizer)


    def get_cumm_noise(effi_noise):
        if FLAGS.dp_sag == False or noise_multiplier == 0:
            return lambda: [torch.Tensor([0]).to(device)] * len(shapes)  # just return scalar 0
        if not effi_noise:
            # we divide num_batches as the sum is over 1./(num_batches).
            cumm_noise = CummuNoiseTorch(np.log(num_batches)*np.sqrt(2)*noise_multiplier * clip / (batch*num_batches), shapes, device)
        else:
            cumm_noise = CummuNoiseEffTorch(np.log(num_batches)*np.sqrt(2)*noise_multiplier * clip / (batch*num_batches), shapes, device)
        return cumm_noise

    cumm_noise = get_cumm_noise(FLAGS.effi_noise)


    # The training loop.
    for epoch in range(1, epochs):
        train_loop(model, device, prev_grad, optimizer, cumm_noise, epoch, writer)

        if epoch + 1 == epochs:
            break
        restart_now = epoch < epochs - 1 and FLAGS.restart > 0 and (epoch + 1) % FLAGS.restart == 0
        if restart_now:
            last_noise = None
            if FLAGS.tree_completion:
                actual_steps = num_batches * FLAGS.restart
                next_pow_2 = 2**(actual_steps - 1).bit_length()
                if next_pow_2 > actual_steps:
                    last_noise = cumm_noise.proceed_until(next_pow_2)
            optimizer.restart(last_noise)
            cumm_noise = get_cumm_noise(FLAGS.effi_noise)
            data_stream.shuffle()  # shuffle the data only when restart
    writer.close()



if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)
