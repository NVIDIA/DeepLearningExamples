#!/usr/bin/env python
"""Example: train LeNet on MNIST (with multi-GPU)."""
from __future__ import division, print_function

import argparse
import functools
import logging
import os.path

from caffe2.python import brew, core, data_parallel_model, optimizer, workspace
from caffe2.python.model_helper import ModelHelper


logging.basicConfig()

TRAIN_ENTRIES = 60000
TEST_ENTRIES = 10000
BATCH_SIZE = 100
EPOCHS = 4
DISPLAY = 100
ACCURACY_MIN = 0.98
ACCURACY_MAX = 0.999


def AddInputOps(model, reader, batch_size):
    """Add input ops."""
    data, label = brew.image_input(
        model, [reader], ['data', 'label'],
        batch_size=batch_size, use_caffe_datum=False, use_gpu_transform=True,
        scale=28, crop=28, mirror=False, color=False, mean=128.0, std=256.0,
        is_test=True)
    data = model.StopGradient(data, data)


def AddForwardPassOps(model, loss_scale):
    """Add forward pass ops and return a list of losses."""
    conv1 = brew.conv(model, 'data', 'conv1', 1, 20, 5)
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    conv2 = brew.conv(model, pool1, 'conv2', 20, 50, 5)
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    fc3 = brew.fc(model, pool2, 'fc3', 50 * 4 * 4, 500)
    fc3 = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, fc3, 'pred', 500, 10)
    softmax, loss = model.SoftmaxWithLoss([pred, 'label'], ['softmax', 'loss'])
    loss = model.Scale(loss, loss, scale=loss_scale)
    brew.accuracy(model, [softmax, 'label'], 'accuracy')
    return [loss]


def AddOptimizerOps(model):
    """Add optimizer ops."""
    optimizer.build_sgd(model, 0.01,
                        policy='step', stepsize=1, gamma=0.999,
                        momentum=0.9, nesterov=False)


def createTrainModel(lmdb_path, devices):
    """Create and return a training model, complete with training ops."""
    model = ModelHelper(name='train', arg_scope={'order': 'NCHW'})
    reader = model.CreateDB('train_reader', db=lmdb_path, db_type='lmdb')
    data_parallel_model.Parallelize_GPU(
        model,
        input_builder_fun=functools.partial(
            AddInputOps, reader=reader,
            batch_size=(BATCH_SIZE // len(devices))),
        forward_pass_builder_fun=AddForwardPassOps,
        optimizer_builder_fun=AddOptimizerOps,
        devices=devices, use_nccl=True)
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    return model


def createTestModel(lmdb_path, devices):
    """Create and return a test model. Does not include training ops."""
    model = ModelHelper(name='test', arg_scope={'order': 'NCHW'},
                        init_params=False)
    reader = model.CreateDB('test_reader', db=lmdb_path, db_type='lmdb')
    data_parallel_model.Parallelize_GPU(
        model,
        input_builder_fun=functools.partial(
            AddInputOps, reader=reader,
            batch_size=(BATCH_SIZE // len(devices))),
        forward_pass_builder_fun=AddForwardPassOps,
        param_update_builder_fun=None,
        devices=devices)
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    return model


def getArgs():
    """Return command-line arguments."""
    CURDIR = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-lmdb', help='Path to training LMDB',
                        default=os.path.join(CURDIR, 'mnist_train_lmdb'))
    parser.add_argument('--test-lmdb', help='Path to test LMDB',
                        default=os.path.join(CURDIR, 'mnist_test_lmdb'))
    parser.add_argument('--gpus',
                        help='Comma separated list of GPU devices to use')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPU devices (instead of --gpus)')
    args = parser.parse_args()

    if args.gpus is not None:
        args.gpus = [int(x) for x in args.gpus.split(',')]
        args.num_gpus = len(args.gpus)
    else:
        args.gpus = range(args.num_gpus)
        args.num_gpus = args.num_gpus
    return args


def main(args):
    """Train and test."""
    train_model = createTrainModel(args.train_lmdb, args.gpus)
    test_model = createTestModel(args.test_lmdb, args.gpus)

    train_iter_per_epoch = TRAIN_ENTRIES // BATCH_SIZE
    test_iter_per_epoch = TEST_ENTRIES // BATCH_SIZE
    scope_prefix = 'gpu_%d/' % args.gpus[0]

    for epoch in range(1, EPOCHS + 1):
        # Train
        for iteration in range(1, train_iter_per_epoch + 1):
            workspace.RunNet(train_model.net.Proto().name)
            if not iteration % DISPLAY:
                loss = workspace.FetchBlob(scope_prefix + 'loss')
                print("Epoch %d/%d, iteration %4d/%d, loss=%f" % (
                    epoch, EPOCHS, iteration, train_iter_per_epoch, loss))

        # Test
        losses = []
        accuracies = []
        for _ in range(test_iter_per_epoch):
            workspace.RunNet(test_model.net.Proto().name)
            # Take average values across all GPUs
            losses.append(sum(
                workspace.FetchBlob('gpu_%d/loss' % g) for g in args.gpus
            ) / len(args.gpus))
            accuracies.append(sum(
                workspace.FetchBlob('gpu_%d/accuracy' % g) for g in args.gpus
            ) / len(args.gpus))

        loss = sum(losses) / len(losses)
        accuracy = sum(accuracies) / len(accuracies)
        print("Test loss: %f, accuracy: %f" % (loss, accuracy))

    if accuracy < ACCURACY_MIN or accuracy > ACCURACY_MAX:
        raise RuntimeError(
            "Final accuracy %f is not in the expected range [%f, %f]" %
            (accuracy, ACCURACY_MIN, ACCURACY_MAX))


if __name__ == '__main__':
    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    main(getArgs())
