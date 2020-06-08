# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import datetime
import os
import numpy as np
import json
from pprint import pprint
from time import time
from sklearn.metrics import roc_auc_score

from absl import app
from absl import flags

import dllogger

import torch
from apex import amp

from dlrm.data import data_loader
from dlrm.data.synthetic_dataset import SyntheticDataset
from dlrm.model import Dlrm

import dlrm.scripts.utils as utils

FLAGS = flags.FLAGS

# Basic run settings
flags.DEFINE_enum("mode", default='train', enum_values=['train', 'test', 'inference_benchmark'],
                  help="Select task to be performed")

flags.DEFINE_integer("seed", 12345, "Random seed")

# Training schedule flags
flags.DEFINE_integer("batch_size", 32768, "Batch size used for training")
flags.DEFINE_integer("test_batch_size", 32768, "Batch size used for testing/validation")
flags.DEFINE_float("lr", 28, "Base learning rate")
flags.DEFINE_integer("epochs", 1, "Number of epochs to train for")
flags.DEFINE_integer("max_steps", None, "Stop training after doing this many optimization steps")

flags.DEFINE_integer("warmup_factor", 0, "Learning rate warmup factor. Must be a non-negative integer")
flags.DEFINE_integer("warmup_steps", 6400, "Number of warmup optimization steps")
flags.DEFINE_integer("decay_steps", 80000, "Polynomial learning rate decay steps. If equal to 0 will not do any decaying")
flags.DEFINE_integer("decay_start_step", 64000,
    "Optimization step after which to start decaying the learning rate, if None will start decaying right after the warmup phase is completed")

# Model configuration
flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of embedding space for categorical features")
flags.DEFINE_list("top_mlp_sizes", [1024, 1024, 512, 256, 1], "Linear layer sizes for the top MLP")
flags.DEFINE_list("bottom_mlp_sizes", [512, 256, 128], "Linear layer sizes for the bottom MLP")

flags.DEFINE_string("interaction_op", "dot",
                    "Type of interaction operation to perform. Supported choices: 'dot' or 'cat'")
flags.DEFINE_boolean("self_interaction", False, "Set to True to use self-interaction")

flags.DEFINE_string(
    "dataset", None,
    "Full path to binary dataset. Must include files such as: train_data.bin, test_data.bin")

flags.DEFINE_boolean("synthetic_dataset", False, "Use synthetic instead of real data for benchmarking purposes")
flags.DEFINE_list("synthetic_dataset_table_sizes", default=','.join(26 * [str(10**5)]),
                  help="Embedding table sizes to use with the synthetic dataset")

flags.DEFINE_boolean("shuffle_batch_order", False, "Read batch in train dataset by random order", short_name="shuffle")

flags.DEFINE_integer("num_numerical_features", 13,
                     "Number of numerical features in the dataset. Defaults to 13 for the Criteo Terabyte Dataset")

flags.DEFINE_integer("max_table_size", None,
                     "Maximum number of rows per embedding table, by default equal to the number of unique values for each categorical variable")
flags.DEFINE_boolean("hash_indices", False,
                     "If True the model will compute `index := index % table size` to ensure that the indices match table sizes")

flags.DEFINE_float("dataset_subset", None,
     "Use only a subset of the training data. If None (default) will use all of it. Must be either None, or a float in range [0,1]")

# Checkpointing
flags.DEFINE_string("load_checkpoint_path", None, "Path from which to load a checkpoint")
flags.DEFINE_string("save_checkpoint_path", None, "Path to which to save the training checkpoints")

# Saving and logging flags
flags.DEFINE_string("output_dir", "/tmp", "Path where to save the checkpoints")
flags.DEFINE_string("log_path", "./log.json", "Destination for the log file with various results and statistics")
flags.DEFINE_integer("test_freq", None, "Number of optimization steps between validations. If None will test after each epoch")
flags.DEFINE_float("test_after", 0, "Don't test the model unless this many epochs has been completed")
flags.DEFINE_integer("print_freq", 200, "Number of optimizations steps between printing training status to stdout")

flags.DEFINE_integer("benchmark_warmup_steps", 0, "Number of initial iterations to exclude from throughput measurements")

# Machine setting flags
flags.DEFINE_string("base_device", "cuda", "Device to run the majority of the model operations")
flags.DEFINE_boolean("fp16", True, "If True (default) the script will use Automatic Mixed Precision")
flags.DEFINE_float("loss_scale", 8192, "Static loss scale for Mixed Precision Training")

# inference benchmark
flags.DEFINE_list("inference_benchmark_batch_sizes", default=[1, 64, 4096],
                  help="Batch sizes for inference throughput and latency measurements")
flags.DEFINE_integer("inference_benchmark_steps", 200,
                     "Number of steps for measuring inference latency and throughput")

flags.DEFINE_float("auc_threshold", None, "Stop the training after achieving this AUC")


def validate_flags():
    if FLAGS.max_table_size is not None and not FLAGS.hash_indices:
       raise ValueError('Hash indices must be True when setting a max_table_size')


def create_synthetic_datasets(train_batch_size, test_batch_size):
    categorical_sizes = get_categorical_feature_sizes()

    dataset_train = SyntheticDataset(num_entries=4 * 10**9,
                                     batch_size=train_batch_size,
                                     dense_features=FLAGS.num_numerical_features,
                                     categorical_feature_sizes=categorical_sizes)

    dataset_test = SyntheticDataset(num_entries=100 * 10**6,
                                    batch_size=test_batch_size,
                                    dense_features=FLAGS.num_numerical_features,
                                    categorical_feature_sizes=categorical_sizes)

    return dataset_train, dataset_test


def create_real_datasets(train_batch_size, test_batch_size, online_shuffle=True):
    train_dataset = os.path.join(FLAGS.dataset, "train_data.bin")
    test_dataset = os.path.join(FLAGS.dataset, "test_data.bin")
    categorical_sizes = get_categorical_feature_sizes()

    dataset_train = data_loader.CriteoBinDataset(
        data_file=train_dataset,
        batch_size=train_batch_size, subset=FLAGS.dataset_subset,
        numerical_features=FLAGS.num_numerical_features,
        categorical_features=len(categorical_sizes),
        online_shuffle=online_shuffle
    )

    dataset_test = data_loader.CriteoBinDataset(
        data_file=test_dataset, batch_size=test_batch_size,
        numerical_features=FLAGS.num_numerical_features,
        categorical_features=len(categorical_sizes),
        online_shuffle = False
    )

    return dataset_train, dataset_test

def get_dataloaders(train_batch_size, test_batch_size):
    print("Creating data loaders")
    if FLAGS.synthetic_dataset:
        dataset_train, dataset_test = create_synthetic_datasets(train_batch_size, test_batch_size)
    else:
        dataset_train, dataset_test = create_real_datasets(train_batch_size,
                                                           test_batch_size,
                                                           online_shuffle=FLAGS.shuffle_batch_order)

    if FLAGS.shuffle_batch_order and not FLAGS.synthetic_dataset:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
    else:
        train_sampler = None
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=None, num_workers=0, pin_memory=False, sampler=train_sampler)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=None, num_workers=0, pin_memory=False)

    return data_loader_train, data_loader_test


def get_categorical_feature_sizes():
    if FLAGS.synthetic_dataset:
        feature_sizes = [int(s) for s in FLAGS.synthetic_dataset_table_sizes]
        return feature_sizes

    categorical_sizes_file = os.path.join(FLAGS.dataset, "model_size.json")
    with open(categorical_sizes_file) as f:
        categorical_sizes = json.load(f).values()

    categorical_sizes = list(categorical_sizes)

    # need to add 1 because the JSON file contains the max value not the count
    categorical_sizes = [s + 1 for s in categorical_sizes]

    if FLAGS.max_table_size is None:
        return categorical_sizes

    clipped_sizes = [min(s, FLAGS.max_table_size) for s in categorical_sizes]
    return clipped_sizes

def create_model():
    print("Creating model")

    model_config = {
        'top_mlp_sizes': FLAGS.top_mlp_sizes,
        'bottom_mlp_sizes': FLAGS.bottom_mlp_sizes,
        'embedding_dim': FLAGS.embedding_dim,
        'interaction_op': FLAGS.interaction_op,
        'self_interaction': FLAGS.self_interaction,
        'categorical_feature_sizes': get_categorical_feature_sizes(),
        'num_numerical_features': FLAGS.num_numerical_features,
        'hash_indices': FLAGS.hash_indices,
        'base_device': FLAGS.base_device,
    }

    model = Dlrm.from_dict(model_config)
    print(model)

    if FLAGS.load_checkpoint_path is not None:
        model.load_state_dict(torch.load(FLAGS.load_checkpoint_path, map_location="cpu"))

    model.to(FLAGS.base_device)

    return model


def main(argv):
    validate_flags()
    torch.manual_seed(FLAGS.seed)

    utils.init_logging(log_path=FLAGS.log_path)
    dllogger.log(data=FLAGS.flag_values_dict(), step='PARAMETER')

    data_loader_train, data_loader_test = get_dataloaders(train_batch_size=FLAGS.batch_size,
                                                          test_batch_size=FLAGS.test_batch_size)

    scaled_lr = FLAGS.lr / FLAGS.loss_scale if FLAGS.fp16 else FLAGS.lr

    model = create_model()

    optimizer = torch.optim.SGD(model.parameters(), lr=scaled_lr)

    if FLAGS.fp16 and FLAGS.mode == 'train':
        (model.top_mlp, model.bottom_mlp), optimizer = amp.initialize([model.top_mlp, model.bottom_mlp],
                                                                      optimizer, opt_level="O2",
                                                                      loss_scale=1)
    elif FLAGS.fp16:
        model = model.half()

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss_fn = torch.jit.trace(loss_fn.forward, (torch.rand(FLAGS.batch_size, 1).cuda(),
                                                torch.rand(FLAGS.batch_size, 1).cuda()))

    if FLAGS.mode == 'test':
        loss, auc, test_step_time = evaluate(model, loss_fn, data_loader_test)

        avg_test_throughput = FLAGS.batch_size / test_step_time
        results = {'auc': auc,
                   'avg_inference_latency': test_step_time,
                   'average_test_throughput': avg_test_throughput}
        dllogger.log(data=results, step=tuple())

        print(F"Finished testing. Test Loss {loss:.4f}, auc {auc:.4f}")
        return

    if FLAGS.mode == 'inference_benchmark':
        results = {}

        if FLAGS.fp16:
            # can use pure FP16 for inference
            model = model.half()

        for batch_size in FLAGS.inference_benchmark_batch_sizes:
            batch_size = int(batch_size)
            _, benchmark_data_loader = get_dataloaders(train_batch_size=batch_size,
                                                       test_batch_size=batch_size)

            latencies = inference_benchmark(model=model, data_loader=benchmark_data_loader,
                                            num_batches=FLAGS.inference_benchmark_steps)

            print("All inference latencies: {}".format(latencies))

            mean_latency = np.mean(latencies)
            mean_inference_throughput = batch_size / mean_latency
            subresult = {F'mean_inference_latency_batch_{batch_size}': mean_latency,
                         F'mean_inference_throughput_batch_{batch_size}': mean_inference_throughput}
            results.update(subresult)
        dllogger.log(data=results, step=tuple())

        print(F"Finished inference benchmark.")
        return

    if FLAGS.mode == 'train':
        train(model, loss_fn, optimizer, data_loader_train, data_loader_test, scaled_lr)


def maybe_save_checkpoint(model, path):
    if path is None:
        return

    begin = time()
    torch.save(model.state_dict(), path)
    end = time()
    print(f'Checkpoint saving took {end-begin:,.2f} [s]')


def train(model, loss_fn, optimizer, data_loader_train, data_loader_test, scaled_lr):
    """Train and evaluate the model

    Args:
        model (dlrm):
        loss_fn (torch.nn.Module): Loss function
        optimizer (torch.nn.optim):
        data_loader_train (torch.utils.data.DataLoader):
        data_loader_test (torch.utils.data.DataLoader):
    """
    model.train()
    base_device = FLAGS.base_device
    print_freq = FLAGS.print_freq
    steps_per_epoch = len(data_loader_train)

    test_freq = FLAGS.test_freq if FLAGS.test_freq is not None else steps_per_epoch

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=print_freq, fmt='{avg:.4f}'))
    metric_logger.add_meter('step_time', utils.SmoothedValue(window_size=print_freq, fmt='{avg:.6f}'))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    timer = utils.StepTimer()

    best_auc = 0
    best_epoch = 0
    start_time = time()
    for epoch in range(FLAGS.epochs):

        batch_iter = iter(data_loader_train)
        for step in range(len(data_loader_train)):
            timer.click()

            global_step = steps_per_epoch * epoch + step

            numerical_features, categorical_features, click = next(batch_iter)

            categorical_features = categorical_features.to(base_device).to(torch.long)
            numerical_features = numerical_features.to(base_device)
            click = click.to(base_device).to(torch.float32)

            utils.lr_step(optimizer, num_warmup_iter=FLAGS.warmup_steps, current_step=global_step + 1,
                          base_lr=scaled_lr, warmup_factor=FLAGS.warmup_factor,
                          decay_steps=FLAGS.decay_steps, decay_start_step=FLAGS.decay_start_step)

            if FLAGS.max_steps and global_step > FLAGS.max_steps:
                print(F"Reached max global steps of {FLAGS.max_steps}. Stopping.")
                break

            output = model(numerical_features, categorical_features).squeeze().float()

            loss = loss_fn(output, click.squeeze())

            optimizer.zero_grad()
            if FLAGS.fp16:
                loss *= FLAGS.loss_scale
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            loss_value = loss.item()

            if timer.measured is None:
                # first iteration, no step time etc. to print
                continue


            if global_step < FLAGS.benchmark_warmup_steps:
                metric_logger.update(
                    loss=loss_value, lr=optimizer.param_groups[0]["lr"])
            else:
                unscale_factor = FLAGS.loss_scale if FLAGS.fp16 else 1
                metric_logger.update(
                     loss=loss_value / unscale_factor, step_time=timer.measured,
                     lr=optimizer.param_groups[0]["lr"] * unscale_factor
                )

            if step % print_freq == 0 and step > 0:
                if global_step < FLAGS.benchmark_warmup_steps:
                    print(F'Warming up, step [{global_step}/{FLAGS.benchmark_warmup_steps}]')
                    continue

                eta_str = datetime.timedelta(seconds=int(metric_logger.step_time.global_avg * (steps_per_epoch - step)))
                metric_logger.print(
                    header=F"Epoch:[{epoch}/{FLAGS.epochs}] [{step}/{steps_per_epoch}]  eta: {eta_str}")

            if (global_step + 1) % test_freq == 0 and global_step > 0 and global_step / steps_per_epoch >= FLAGS.test_after:
                loss, auc, test_step_time = evaluate(model, loss_fn, data_loader_test)
                print(F"Epoch {epoch} step {step}. Test loss {loss:.5f}, auc {auc:.6f}")

                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch + ((step + 1) / steps_per_epoch)
                    maybe_save_checkpoint(model, FLAGS.save_checkpoint_path)

                if FLAGS.auc_threshold and auc >= FLAGS.auc_threshold:
                    stop_time = time()
                    run_time_s = int(stop_time - start_time)
                    print(F"Hit target accuracy AUC {FLAGS.auc_threshold} at epoch "
                          F"{global_step/steps_per_epoch:.2f} in {run_time_s}s. "
                          F"Average speed {global_step * FLAGS.batch_size / run_time_s:.1f} records/s.")
                    return

    avg_throughput = FLAGS.batch_size / metric_logger.step_time.avg

    results = {'best_auc' : best_auc,
               'best_epoch' : best_epoch,
               'average_train_throughput' : avg_throughput}

    if 'test_step_time' in locals():
        avg_test_throughput = FLAGS.test_batch_size / test_step_time
        results['average_test_throughput'] = avg_test_throughput

    dllogger.log(data=results, step=tuple())


def evaluate(model, loss_fn, data_loader):
    """Test dlrm model

    Args:
        model (dlrm):
        loss_fn (torch.nn.Module): Loss function
        data_loader (torch.utils.data.DataLoader):
    """
    model.eval()
    base_device = FLAGS.base_device
    print_freq = FLAGS.print_freq

    steps_per_epoch = len(data_loader)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=print_freq, fmt='{avg:.4f}'))
    metric_logger.add_meter('step_time', utils.SmoothedValue(window_size=print_freq, fmt='{avg:.4f}'))
    with torch.no_grad():
        y_true = []
        y_score = []

        timer = utils.StepTimer()
        batch_iter = iter(data_loader)

        timer.click()
        for step in range(len(data_loader)):
            numerical_features, categorical_features, click = next(batch_iter)

            categorical_features = categorical_features.to(base_device).to(torch.long)
            numerical_features = numerical_features.to(base_device)
            click = click.to(torch.float32).to(base_device)

            if FLAGS.fp16:
                numerical_features = numerical_features.half()

            output = model(numerical_features, categorical_features).squeeze()

            loss = loss_fn(output, click)
            y_true.append(click)
            y_score.append(output)

            loss_value = loss.item()
            timer.click()

            if timer.measured is not None:
                metric_logger.update(loss=loss_value, step_time=timer.measured)
                if step % print_freq == 0 and step > 0:
                    metric_logger.print(header=F"Test: [{step}/{steps_per_epoch}]")

        y_true = torch.cat(y_true).cpu().numpy()
        y_score = torch.cat(y_score).cpu().numpy()
        auc = roc_auc_score(y_true=y_true, y_score=y_score)

    model.train()

    return metric_logger.loss.global_avg, auc, metric_logger.step_time.avg


def inference_benchmark(model, data_loader, num_batches=100):
    model.eval()
    base_device = FLAGS.base_device
    latencies = []

    with torch.no_grad():
        for step, (numerical_features, categorical_features, click) in enumerate(data_loader):
            if step > num_batches:
                break

            step_start_time = time()

            numerical_features = numerical_features.to(base_device)
            if FLAGS.fp16:
                numerical_features = numerical_features.half()

            categorical_features = categorical_features.to(device=base_device, dtype=torch.int64)

            _ = model(numerical_features, categorical_features).squeeze()
            torch.cuda.synchronize()
            step_time = time() - step_start_time

            if step >= FLAGS.benchmark_warmup_steps:
                latencies.append(step_time)
    return latencies


if __name__ == '__main__':
    app.run(main)

