#!/usr/bin/env python
"""Example: train LeNet on MNIST (with fp16)."""
from __future__ import division, print_function

import argparse
import os.path

import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python import brew, core, optimizer, workspace
from caffe2.python.model_helper import ModelHelper
from caffe2.python.modeling.initializers import pFP16Initializer


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
        output_type='float16', is_test=True)
    data = model.StopGradient(data, data)


def AddForwardPassOps(model):
    """Add forward pass ops and return a list of losses."""
    with brew.arg_scope([brew.conv, brew.fc],
                        WeightInitializer=pFP16Initializer,
                        BiasInitializer=pFP16Initializer):
        conv1 = brew.conv(model, 'data', 'conv1', 1, 20, 5)
        pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
        conv2 = brew.conv(model, pool1, 'conv2', 20, 50, 5)
        pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
        fc3 = brew.fc(model, pool2, 'fc3', 50 * 4 * 4, 500)
        fc3 = brew.relu(model, fc3, fc3)
        pred = brew.fc(model, fc3, 'pred', 500, 10)

    # Cast back to fp32 for remaining ops
    pred = model.net.HalfToFloat(pred, pred + '_fp32')
    softmax, loss = model.SoftmaxWithLoss([pred, 'label'], ['softmax', 'loss'])
    brew.accuracy(model, [softmax, 'label'], 'accuracy')
    return [loss]


def AddOptimizerOps(model):
    """Add optimizer ops."""
    optimizer.build_sgd(model, 0.01,
                        policy='step', stepsize=1, gamma=0.999,
                        momentum=0.9, nesterov=False)


def createTrainModel(lmdb_path):
    """Create and return a training model, complete with training ops."""
    model = ModelHelper(name='train', arg_scope={'order': 'NCHW'})
    reader = model.CreateDB('train_reader', db=lmdb_path, db_type='lmdb')
    AddInputOps(model, reader, BATCH_SIZE)
    losses = AddForwardPassOps(model)
    model.AddGradientOperators(losses)
    AddOptimizerOps(model)
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    return model


def createTestModel(lmdb_path):
    """Create and return a test model. Does not include training ops."""
    model = ModelHelper(name='test', arg_scope={'order': 'NCHW'},
                        init_params=False)
    reader = model.CreateDB('test_reader', db=lmdb_path, db_type='lmdb')
    AddInputOps(model, reader, BATCH_SIZE)
    AddForwardPassOps(model)
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
    args = parser.parse_args()
    return args


def main(args):
    """Train and test."""
    device = 0
    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, device)):
        train_model = createTrainModel(args.train_lmdb)
        test_model = createTestModel(args.test_lmdb)

    train_iter_per_epoch = TRAIN_ENTRIES // BATCH_SIZE
    test_iter_per_epoch = TEST_ENTRIES // BATCH_SIZE

    for epoch in range(1, EPOCHS + 1):
        # Train
        for iteration in range(1, train_iter_per_epoch + 1):
            workspace.RunNet(train_model.net.Proto().name)
            if not iteration % DISPLAY:
                loss = workspace.FetchBlob('loss')
                print("Epoch %d/%d, iteration %4d/%d, loss=%f" % (
                    epoch, EPOCHS, iteration, train_iter_per_epoch, loss))

        # Test
        losses = []
        accuracies = []
        for _ in range(test_iter_per_epoch):
            workspace.RunNet(test_model.net.Proto().name)
            losses.append(workspace.FetchBlob('loss'))
            accuracies.append(workspace.FetchBlob('accuracy'))

        loss = np.array(losses).mean()
        accuracy = np.array(accuracies).mean()
        print("Test loss: %f, accuracy: %f" % (loss, accuracy))

    if accuracy < ACCURACY_MIN or accuracy > ACCURACY_MAX:
        raise RuntimeError(
            "Final accuracy %f is not in the expected range [%f, %f]" %
            (accuracy, ACCURACY_MIN, ACCURACY_MAX))


if __name__ == '__main__':
    core.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    main(getArgs())
