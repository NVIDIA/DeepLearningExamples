#!/usr/bin/env python
"""Example: train a model on CIFAR10."""
from __future__ import division, print_function

import argparse
import functools
import logging
import os.path

from caffe2.python import brew, core, data_parallel_model, optimizer, workspace
from caffe2.python.core import DataType
from caffe2.python.model_helper import ModelHelper
from caffe2.python.modeling.initializers import Initializer, pFP16Initializer


logging.basicConfig()

TRAIN_ENTRIES = 50000
TEST_ENTRIES = 10000
BATCH_SIZE = 100
EPOCHS = 10
DISPLAY = 100
ACCURACY_MIN = 0.7
ACCURACY_MAX = 0.8


def AddInputOps(model, reader, batch_size, dtype):
    """Add input ops."""
    data, label = brew.image_input(
        model, [reader], ['data', 'label'],
        batch_size=batch_size, use_caffe_datum=False, use_gpu_transform=True,
        scale=32, crop=32, mirror=1, color=True, mean=128.0,
        output_type='float16' if dtype == DataType.FLOAT16 else 'float',
        is_test=False)
    data = model.StopGradient(data, data)


def AddForwardPassOps(model, loss_scale, dtype):
    """Add forward pass ops and return a list of losses."""
    initializer = (pFP16Initializer if dtype == DataType.FLOAT16
                   else Initializer)
    with brew.arg_scope([brew.conv, brew.fc],
                        WeightInitializer=initializer,
                        BiasInitializer=initializer):
        conv1 = brew.conv(model, 'data', 'conv1', 3, 32, 5, pad=2,
                          weight_init=('GaussianFill',
                                       {'std': 0.0001, 'mean': 0.0}))
        pool1 = brew.max_pool(model, conv1, 'pool1', kernel=3, stride=2)
        relu1 = brew.relu(model, pool1, 'relu1')
        conv2 = brew.conv(model, relu1, 'conv2', 32, 32, 5, pad=2,
                          weight_init=('GaussianFill', {'std': 0.01}))
        conv2 = brew.relu(model, conv2, conv2)
        pool2 = brew.average_pool(model, conv2, 'pool2', kernel=3, stride=2)
        conv3 = brew.conv(model, pool2, 'conv3', 32, 64, 5, pad=2,
                          weight_init=('GaussianFill', {'std': 0.01}))
        conv3 = brew.relu(model, conv3, conv3)
        pool3 = brew.average_pool(model, conv3, 'pool3', kernel=3, stride=2)
        fc1 = brew.fc(model, pool3, 'fc1', 64 * 3 * 3, 64,
                      weight_init=('GaussianFill', {'std': 0.1}))
        fc2 = brew.fc(model, fc1, 'fc2', 64, 10,
                      weight_init=('GaussianFill', {'std': 0.1}))

    if dtype == DataType.FLOAT16:
        fc2 = model.net.HalfToFloat(fc2, fc2 + '_fp32')
    softmax, loss = model.SoftmaxWithLoss([fc2, 'label'], ['softmax', 'loss'])
    loss = model.Scale(loss, loss, scale=loss_scale)
    brew.accuracy(model, [softmax, 'label'], 'accuracy')
    return [loss]


def AddOptimizerOps(model):
    """Add optimizer ops."""
    optimizer.add_weight_decay(model, 0.004)
    stepsize = TRAIN_ENTRIES * EPOCHS // BATCH_SIZE
    optimizer.build_sgd(
        model, 0.001,
        policy='step', stepsize=stepsize, gamma=0.1,
        momentum=0.9, nesterov=False)


def AddPostSyncOps(model):
    """Add ops which run after the initial parameter sync."""
    for param_info in model.GetOptimizationParamInfo(model.GetParams()):
        if param_info.blob_copy is not None:
            # Ensure copies are in sync after initial broadcast
            model.param_init_net.HalfToFloat(
                param_info.blob,
                param_info.blob_copy[core.DataType.FLOAT]
            )


def createTrainModel(lmdb_path, devices, dtype):
    """Create and return a training model, complete with training ops."""
    model = ModelHelper(name='train', arg_scope={'order': 'NCHW'})
    reader = model.CreateDB('train_reader', db=lmdb_path, db_type='lmdb')
    data_parallel_model.Parallelize_GPU(
        model,
        input_builder_fun=functools.partial(
            AddInputOps, reader=reader,
            batch_size=(BATCH_SIZE // len(devices)), dtype=dtype),
        forward_pass_builder_fun=functools.partial(
            AddForwardPassOps, dtype=dtype),
        optimizer_builder_fun=AddOptimizerOps,
        post_sync_builder_fun=AddPostSyncOps,
        devices=devices, use_nccl=True)
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    return model


def createTestModel(lmdb_path, devices, dtype):
    """Create and return a test model. Does not include training ops."""
    model = ModelHelper(name='test', arg_scope={'order': 'NCHW'},
                        init_params=False)
    reader = model.CreateDB('test_reader', db=lmdb_path, db_type='lmdb')
    data_parallel_model.Parallelize_GPU(
        model,
        input_builder_fun=functools.partial(
            AddInputOps, reader=reader,
            batch_size=(BATCH_SIZE // len(devices)), dtype=dtype),
        forward_pass_builder_fun=functools.partial(
            AddForwardPassOps, dtype=dtype),
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
                        default=os.path.join(CURDIR, 'cifar10_train_lmdb'))
    parser.add_argument('--test-lmdb', help='Path to test LMDB',
                        default=os.path.join(CURDIR, 'cifar10_test_lmdb'))
    parser.add_argument('--dtype', choices=['float', 'float16'],
                        default='float', help='Data type used for training')
    parser.add_argument('--gpus',
                        help='Comma separated list of GPU devices to use')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPU devices (instead of --gpus)')
    parser.add_argument('--all-gpus', action='store_true',
                        help='Use all GPUs in the system')
    args = parser.parse_args()

    args.dtype = (DataType.FLOAT16 if args.dtype == 'float16'
                  else DataType.FLOAT)

    if args.all_gpus:
        args.num_gpus = workspace.NumCudaDevices()
        args.gpus = range(args.num_gpus)
    else:
        if args.gpus is not None:
            args.gpus = [int(x) for x in args.gpus.split(',')]
            args.num_gpus = len(args.gpus)
        else:
            args.gpus = range(args.num_gpus)
            args.num_gpus = args.num_gpus
    return args


def main(args):
    """Train and test."""
    train_model = createTrainModel(args.train_lmdb, args.gpus, args.dtype)
    test_model = createTestModel(args.test_lmdb, args.gpus, args.dtype)

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
