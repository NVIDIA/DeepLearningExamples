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
from time import time

import dllogger
import numpy as np
import torch
from absl import app, flags
from apex import amp

import dlrm.scripts.utils as utils
from dlrm.data.data_loader import get_data_loaders
from dlrm.data.utils import get_categorical_feature_sizes, prefetcher
from dlrm.model.single import Dlrm
from dlrm.utils.checkpointing.serial import SerialCheckpointWriter, make_serial_checkpoint_writer, \
    make_serial_checkpoint_loader

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
flags.DEFINE_integer("decay_power", 2, "Polynomial learning rate decay power")
flags.DEFINE_float("decay_end_lr", 0, "LR after the decay ends")

# Model configuration
flags.DEFINE_enum("embedding_type", "joint_fused", ["joint", "joint_fused", "joint_sparse", "multi_table"],
                  help="The type of the embedding operation to use")
flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of embedding space for categorical features")
flags.DEFINE_list("top_mlp_sizes", [1024, 1024, 512, 256, 1], "Linear layer sizes for the top MLP")
flags.DEFINE_list("bottom_mlp_sizes", [512, 256, 128], "Linear layer sizes for the bottom MLP")

flags.DEFINE_enum("interaction_op", default="cuda_dot", enum_values=["cuda_dot", "dot", "cat"],
                  help="Type of interaction operation to perform.")

flags.DEFINE_string(
    "dataset", None,
    "Full path to binary dataset. Must include files such as: train_data.bin, test_data.bin")
flags.DEFINE_enum("dataset_type", default="split", enum_values=['binary', 'split', 'synthetic_gpu', 'synthetic_disk'],
                  help='The type of the dataset to use')

flags.DEFINE_string("synthetic_dataset_dir", "/tmp/dlrm_sythetic_dataset", "Default synthetic disk dataset directory")
flags.DEFINE_list("synthetic_dataset_table_sizes", default=','.join(26 * [str(10**5)]),
                  help="Embedding table sizes to use with the synthetic dataset")

flags.DEFINE_integer("synthetic_dataset_num_entries", default=int(2**15 * 1024), # 1024 batches by default
                     help="Number of samples per epoch for the synthetic dataset")

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
flags.DEFINE_boolean("amp", False, "If True the script will use Automatic Mixed Precision")
flags.DEFINE_float("loss_scale", 1024, "Static loss scale for Mixed Precision Training")

# inference benchmark
flags.DEFINE_list("inference_benchmark_batch_sizes", default=[1, 64, 4096],
                  help="Batch sizes for inference throughput and latency measurements")
flags.DEFINE_integer("inference_benchmark_steps", 200,
                     "Number of steps for measuring inference latency and throughput")

flags.DEFINE_float("auc_threshold", None, "Stop the training after achieving this AUC")
flags.DEFINE_boolean("optimized_mlp", True, "Use an optimized implementation of MLP from apex")


def validate_flags():
    if FLAGS.max_table_size is not None and not FLAGS.hash_indices:
       raise ValueError('Hash indices must be True when setting a max_table_size')

    if FLAGS.base_device == 'cpu':
        if FLAGS.embedding_type in ('joint_fused', 'joint_sparse'):
            print('WARNING: CUDA joint embeddings are not supported on CPU')
            FLAGS.embedding_type = 'joint'

        if FLAGS.amp:
            print('WARNING: Automatic mixed precision not supported on CPU')
            FLAGS.amp = False

        if FLAGS.optimized_mlp:
            print('WARNING: Optimized MLP is not supported on CPU')
            FLAGS.optimized_mlp = False


def is_data_prefetching_enabled() -> bool:
    return FLAGS.base_device == 'cuda'


def create_model():
    print("Creating model")

    model_config = {
        'top_mlp_sizes': FLAGS.top_mlp_sizes,
        'bottom_mlp_sizes': FLAGS.bottom_mlp_sizes,
        'embedding_type': FLAGS.embedding_type,
        'embedding_dim': FLAGS.embedding_dim,
        'interaction_op': FLAGS.interaction_op,
        'categorical_feature_sizes': get_categorical_feature_sizes(FLAGS),
        'num_numerical_features': FLAGS.num_numerical_features,
        'hash_indices': FLAGS.hash_indices,
        'use_cpp_mlp': FLAGS.optimized_mlp,
        'fp16': FLAGS.amp,
        'base_device': FLAGS.base_device,
    }

    model = Dlrm.from_dict(model_config)
    print(model)

    model.to(FLAGS.base_device)

    if FLAGS.load_checkpoint_path is not None:
        checkpoint_loader = make_serial_checkpoint_loader(
            embedding_indices=range(len(get_categorical_feature_sizes(FLAGS))),
            device="cpu"
        )
        checkpoint_loader.load_checkpoint(model, FLAGS.load_checkpoint_path)
        model.to(FLAGS.base_device)

    return model


def main(argv):
    validate_flags()
    torch.manual_seed(FLAGS.seed)

    utils.init_logging(log_path=FLAGS.log_path)
    dllogger.log(data=FLAGS.flag_values_dict(), step='PARAMETER')

    data_loader_train, data_loader_test = get_data_loaders(FLAGS)

    scaled_lr = FLAGS.lr / FLAGS.loss_scale if FLAGS.amp else FLAGS.lr

    model = create_model()

    optimizer = torch.optim.SGD(model.parameters(), lr=scaled_lr)

    if FLAGS.amp and FLAGS.mode == 'train':
        (model.top_model, model.bottom_model.mlp), optimizer = amp.initialize([model.top_model, model.bottom_model.mlp],
                                                                              optimizer, opt_level="O2", loss_scale=1)
    elif FLAGS.amp:
        model = model.half()

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

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

        if FLAGS.amp:
            # can use pure FP16 for inference
            model = model.half()

        for batch_size in FLAGS.inference_benchmark_batch_sizes:
            batch_size = int(batch_size)
            FLAGS.test_batch_size = batch_size

            _, benchmark_data_loader = get_data_loaders(FLAGS)

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


def maybe_save_checkpoint(checkpoint_writer: SerialCheckpointWriter, model, path):
    if path is None:
        return

    print(f'Saving a checkpoint to {path}')

    begin = time()
    checkpoint_writer.save_checkpoint(model, path)
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
    prefetching_enabled = is_data_prefetching_enabled()
    base_device = FLAGS.base_device
    print_freq = FLAGS.print_freq
    steps_per_epoch = len(data_loader_train)

    checkpoint_writer = make_serial_checkpoint_writer(
        embedding_indices=range(len(get_categorical_feature_sizes(FLAGS))),
        config=FLAGS.flag_values_dict()
    )

    test_freq = FLAGS.test_freq if FLAGS.test_freq is not None else steps_per_epoch - 1

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('step_time', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    if prefetching_enabled:
        data_stream = torch.cuda.Stream()

    timer = utils.StepTimer()

    best_auc = 0
    best_epoch = 0
    start_time = time()

    timer.click()

    for epoch in range(FLAGS.epochs):
        input_pipeline = iter(data_loader_train)

        if prefetching_enabled:
            input_pipeline = prefetcher(input_pipeline, data_stream)

        for step, batch in enumerate(input_pipeline):
            global_step = steps_per_epoch * epoch + step
            numerical_features, categorical_features, click = batch

            utils.lr_step(optimizer, num_warmup_iter=FLAGS.warmup_steps, current_step=global_step + 1,
                          base_lr=scaled_lr, warmup_factor=FLAGS.warmup_factor,
                          decay_steps=FLAGS.decay_steps, decay_start_step=FLAGS.decay_start_step)

            if FLAGS.max_steps and global_step > FLAGS.max_steps:
                print(F"Reached max global steps of {FLAGS.max_steps}. Stopping.")
                break

            if prefetching_enabled:
                torch.cuda.synchronize()

            output = model(numerical_features, categorical_features).squeeze().float()

            loss = loss_fn(output, click.squeeze())

            # Setting grad to None is faster than zero_grad()
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    param.grad = None

            if FLAGS.amp:
                loss *= FLAGS.loss_scale
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            if step % print_freq == 0 and step > 0:
                loss_value = loss.item()

                timer.click()

                if global_step < FLAGS.benchmark_warmup_steps:
                    metric_logger.update(
                        loss=loss_value, lr=optimizer.param_groups[0]["lr"])
                else:
                    unscale_factor = FLAGS.loss_scale if FLAGS.amp else 1
                    metric_logger.update(
                        loss=loss_value / unscale_factor,
                        step_time=timer.measured / FLAGS.print_freq,
                        lr=optimizer.param_groups[0]["lr"] * unscale_factor
                    )

                if global_step < FLAGS.benchmark_warmup_steps:
                    print(F'Warming up, step [{global_step}/{FLAGS.benchmark_warmup_steps}]')
                    continue

                eta_str = datetime.timedelta(seconds=int(metric_logger.step_time.global_avg * (steps_per_epoch - step)))
                metric_logger.print(
                    header=F"Epoch:[{epoch}/{FLAGS.epochs}] [{step}/{steps_per_epoch}]  eta: {eta_str}")

            if (global_step % test_freq == 0 and global_step > 0 and
                    global_step / steps_per_epoch >= FLAGS.test_after):
                loss, auc, test_step_time = evaluate(model, loss_fn, data_loader_test)
                print(F"Epoch {epoch} step {step}. Test loss {loss:.5f}, auc {auc:.6f}")

                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch + ((step + 1) / steps_per_epoch)
                    maybe_save_checkpoint(checkpoint_writer, model, FLAGS.save_checkpoint_path)

                if FLAGS.auc_threshold and auc >= FLAGS.auc_threshold:
                    stop_time = time()
                    run_time_s = int(stop_time - start_time)
                    print(F"Hit target accuracy AUC {FLAGS.auc_threshold} at epoch "
                          F"{global_step/steps_per_epoch:.2f} in {run_time_s}s. "
                          F"Average speed {global_step * FLAGS.batch_size / run_time_s:.1f} records/s.")
                    return

    stop_time = time()
    run_time_s = int(stop_time - start_time)

    print(F"Finished training in {run_time_s}s. "
          F"Average speed {global_step * FLAGS.batch_size / run_time_s:.1f} records/s.")

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
    print_freq = FLAGS.print_freq
    prefetching_enabled = is_data_prefetching_enabled()

    steps_per_epoch = len(data_loader)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{avg:.4f}'))
    metric_logger.add_meter('step_time', utils.SmoothedValue(window_size=1, fmt='{avg:.4f}'))

    if prefetching_enabled:
        data_stream = torch.cuda.Stream()

    with torch.no_grad():
        y_true = []
        y_score = []

        timer = utils.StepTimer()
        timer.click()

        input_pipeline = iter(data_loader)

        if prefetching_enabled:
            input_pipeline = prefetcher(input_pipeline, data_stream)

        for step, (numerical_features, categorical_features, click) in enumerate(input_pipeline):
            if FLAGS.amp:
                numerical_features = numerical_features.half()

            if prefetching_enabled:
                torch.cuda.synchronize()

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

        y_true = torch.cat(y_true)
        y_score = torch.cat(y_score)

        before_auc_timestamp = time()
        auc = utils.roc_auc_score(y_true=y_true, y_score=y_score)
        print(f'AUC computation took: {time() - before_auc_timestamp:.2f} [s]')

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
            if FLAGS.amp:
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

