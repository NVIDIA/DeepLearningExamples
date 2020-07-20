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
import itertools
import sys
from pprint import pprint
from time import time

import dllogger
import numpy as np
import torch
from absl import app, flags, logging
from apex import amp, parallel, optimizers as apex_optim

import dlrm.scripts.utils as utils
from dlrm.data.data_loader import get_data_loaders
from dlrm.data.utils import prefetcher
from dlrm.model.distributed import DistributedDlrm
from dlrm.scripts.main import FLAGS, get_categorical_feature_sizes
from dlrm.utils import distributed as dist
from dlrm.utils.checkpointing.distributed import make_distributed_checkpoint_writer, make_distributed_checkpoint_loader
from dlrm.utils.distributed import get_gpu_batch_sizes, get_criteo_device_mapping, is_main_process, is_distributed

# Training schedule flags
FLAGS.set_default("batch_size", 65536)
FLAGS.set_default("test_batch_size", 131072)
FLAGS.set_default("lr", 24.0)
FLAGS.set_default("warmup_factor", 0)
FLAGS.set_default("warmup_steps", 8000)
FLAGS.set_default("decay_steps", 24000)
FLAGS.set_default("decay_start_step", 48000)
FLAGS.set_default("decay_power", 2)
FLAGS.set_default("decay_end_lr", 0)
FLAGS.set_default("embedding_type", "joint_sparse")

flags.DEFINE_string("backend", "nccl", "Backend to use for distributed training. Default nccl")
flags.DEFINE_boolean("bottom_features_ordered", False, "Sort features from the bottom model, useful when using saved "
                                                       "checkpoint in different device configurations")


def main(argv):
    torch.manual_seed(FLAGS.seed)

    utils.init_logging(log_path=FLAGS.log_path)

    use_gpu = "cpu" not in FLAGS.base_device.lower()
    rank, world_size, gpu = dist.init_distributed_mode(backend=FLAGS.backend, use_gpu=use_gpu)
    device = FLAGS.base_device

    if not is_distributed():
        raise NotImplementedError("This file is only for distributed training.")

    if is_main_process():
        dllogger.log(data=FLAGS.flag_values_dict(), step='PARAMETER')

        print("Command line flags:")
        pprint(FLAGS.flag_values_dict())

    print("Creating data loaders")

    FLAGS.set_default("test_batch_size", FLAGS.test_batch_size // world_size * world_size)

    categorical_feature_sizes = get_categorical_feature_sizes(FLAGS)
    world_categorical_feature_sizes = np.asarray(categorical_feature_sizes)
    device_mapping = get_criteo_device_mapping(world_size)

    batch_sizes_per_gpu = get_gpu_batch_sizes(FLAGS.batch_size, num_gpus=world_size)
    batch_indices = tuple(np.cumsum([0] + list(batch_sizes_per_gpu)))

    # sizes of embeddings for each GPU
    categorical_feature_sizes = world_categorical_feature_sizes[device_mapping['embedding'][rank]].tolist()

    bottom_mlp_sizes = FLAGS.bottom_mlp_sizes if rank == device_mapping['bottom_mlp'] else None

    data_loader_train, data_loader_test = get_data_loaders(FLAGS, device_mapping=device_mapping)

    model = DistributedDlrm(
        vectors_per_gpu=device_mapping['vectors_per_gpu'],
        embedding_device_mapping=device_mapping['embedding'],
        embedding_type=FLAGS.embedding_type,
        embedding_dim=FLAGS.embedding_dim,
        world_num_categorical_features=len(world_categorical_feature_sizes),
        categorical_feature_sizes=categorical_feature_sizes,
        num_numerical_features=FLAGS.num_numerical_features,
        hash_indices=FLAGS.hash_indices,
        bottom_mlp_sizes=bottom_mlp_sizes,
        top_mlp_sizes=FLAGS.top_mlp_sizes,
        interaction_op=FLAGS.interaction_op,
        fp16=FLAGS.amp,
        use_cpp_mlp=FLAGS.optimized_mlp,
        bottom_features_ordered=FLAGS.bottom_features_ordered,
        device=device
    )
    print(model)
    print(device_mapping)
    print(f"Batch sizes per gpu: {batch_sizes_per_gpu}")

    dist.setup_distributed_print(is_main_process())

    # DDP introduces a gradient average through allreduce(mean), which doesn't apply to bottom model.
    # Compensate it with further scaling lr
    scaled_lr = FLAGS.lr / FLAGS.loss_scale if FLAGS.amp else FLAGS.lr
    scaled_lrs = [scaled_lr / world_size, scaled_lr]

    embedding_optimizer = torch.optim.SGD([
        {'params': model.bottom_model.embeddings.parameters(), 'lr': scaled_lrs[0]},
    ])
    mlp_optimizer = apex_optim.FusedSGD([
        {'params': model.bottom_model.mlp.parameters(), 'lr': scaled_lrs[0]},
        {'params': model.top_model.parameters(), 'lr': scaled_lrs[1]}
    ])

    checkpoint_writer = make_distributed_checkpoint_writer(
        device_mapping=device_mapping,
        rank=rank,
        is_main_process=is_main_process(),
        config=FLAGS.flag_values_dict()
    )

    checkpoint_loader = make_distributed_checkpoint_loader(device_mapping=device_mapping, rank=rank)

    if FLAGS.load_checkpoint_path:
        checkpoint_loader.load_checkpoint(model, FLAGS.load_checkpoint_path)
        model.to(device)

    if FLAGS.amp:
        (model.top_model, model.bottom_model.mlp), mlp_optimizer = amp.initialize(
            [model.top_model, model.bottom_model.mlp], mlp_optimizer, opt_level="O2", loss_scale=1)

    if use_gpu:
        model.top_model = parallel.DistributedDataParallel(model.top_model)
    else:  # Use other backend for CPU
        model.top_model = torch.nn.parallel.DistributedDataParallel(model.top_model)

    if FLAGS.mode == 'test':
        auc = dist_evaluate(model, data_loader_test)

        results = {'auc': auc}
        dllogger.log(data=results, step=tuple())

        if auc is not None:
            print(F"Finished testing. Test auc {auc:.4f}")
        return

    if FLAGS.save_checkpoint_path and not FLAGS.bottom_features_ordered and is_main_process():
        logging.warning("Saving checkpoint without --bottom_features_ordered flag will result in "
                        "a device-order dependent model. Consider using --bottom_features_ordered "
                        "if you plan to load the checkpoint in different device configurations.")

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

    # Print per 16384 * 2000 samples by default
    default_print_freq = 16384 * 2000 // FLAGS.batch_size
    print_freq = default_print_freq if FLAGS.print_freq is None else FLAGS.print_freq

    steps_per_epoch = len(data_loader_train)
    test_freq = FLAGS.test_freq if FLAGS.test_freq is not None else steps_per_epoch - 1

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{avg:.4f}'))
    metric_logger.add_meter('step_time', utils.SmoothedValue(window_size=1, fmt='{avg:.6f}'))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    # Accumulating loss on GPU to avoid memcpyD2H every step
    moving_loss = torch.zeros(1, device=device)
    moving_loss_stream = torch.cuda.Stream()

    lr_scheduler = utils.LearningRateScheduler(optimizers=[mlp_optimizer, embedding_optimizer],
                                               base_lrs=[scaled_lrs, [scaled_lrs[0]]],
                                               warmup_steps=FLAGS.warmup_steps,
                                               warmup_factor=FLAGS.warmup_factor,
                                               decay_start_step=FLAGS.decay_start_step,
                                               decay_steps=FLAGS.decay_steps,
                                               decay_power=FLAGS.decay_power,
                                               end_lr_factor=FLAGS.decay_end_lr / FLAGS.lr)

    data_stream = torch.cuda.Stream()
    timer = utils.StepTimer()

    best_auc = 0
    best_epoch = 0
    start_time = time()
    stop_time = time()

    for epoch in range(FLAGS.epochs):
        epoch_start_time = time()

        batch_iter = prefetcher(iter(data_loader_train), data_stream)

        for step in range(len(data_loader_train)):
            timer.click()

            numerical_features, categorical_features, click = next(batch_iter)
            torch.cuda.synchronize()

            global_step = steps_per_epoch * epoch + step

            if FLAGS.max_steps and global_step > FLAGS.max_steps:
                print(F"Reached max global steps of {FLAGS.max_steps}. Stopping.")
                break

            lr_scheduler.step()

            if click.shape[0] != FLAGS.batch_size:  # last batch
                logging.error("The last batch with size %s is not supported", click.shape[0])
            else:
                output = model(numerical_features, categorical_features, batch_sizes_per_gpu).squeeze()

                loss = loss_fn(output, click[batch_indices[rank]: batch_indices[rank + 1]])

                # We don't need to accumulate gradient. Set grad to None is faster than optimizer.zero_grad()
                for param_group in itertools.chain(embedding_optimizer.param_groups, mlp_optimizer.param_groups):
                    for param in param_group['params']:
                        param.grad = None

                if FLAGS.amp:
                    loss *= FLAGS.loss_scale
                    with amp.scale_loss(loss, mlp_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                mlp_optimizer.step()
                embedding_optimizer.step()

                moving_loss_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(moving_loss_stream):
                    moving_loss += loss

            if timer.measured is None:
                # first iteration, no step time etc. to print
                continue

            if step == 0:
                print(F"Started epoch {epoch}...")
            elif step % print_freq == 0:
                torch.cuda.current_stream().wait_stream(moving_loss_stream)
                # Averaging cross a print_freq period to reduce the error.
                # An accurate timing needs synchronize which would slow things down.

                if global_step < FLAGS.benchmark_warmup_steps:
                    metric_logger.update(
                        loss=moving_loss.item() / print_freq / (FLAGS.loss_scale if FLAGS.amp else 1),
                        lr=mlp_optimizer.param_groups[1]["lr"] * (FLAGS.loss_scale if FLAGS.amp else 1))
                else:
                    metric_logger.update(
                        step_time=timer.measured,
                        loss=moving_loss.item() / print_freq / (FLAGS.loss_scale if FLAGS.amp else 1),
                        lr=mlp_optimizer.param_groups[1]["lr"] * (FLAGS.loss_scale if FLAGS.amp else 1))
                stop_time = time()

                eta_str = datetime.timedelta(seconds=int(metric_logger.step_time.global_avg * (steps_per_epoch - step)))
                metric_logger.print(
                    header=F"Epoch:[{epoch}/{FLAGS.epochs}] [{step}/{steps_per_epoch}]  eta: {eta_str}")

                with torch.cuda.stream(moving_loss_stream):
                    moving_loss = 0.

            if global_step % test_freq == 0 and global_step > 0 and global_step / steps_per_epoch >= FLAGS.test_after:
                auc = dist_evaluate(model, data_loader_test)

                if auc is None:
                    continue

                print(F"Epoch {epoch} step {step}. auc {auc:.6f}")
                stop_time = time()

                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch + ((step + 1) / steps_per_epoch)

                if FLAGS.auc_threshold and auc >= FLAGS.auc_threshold:
                    run_time_s = int(stop_time - start_time)
                    print(F"Hit target accuracy AUC {FLAGS.auc_threshold} at epoch "
                          F"{global_step/steps_per_epoch:.2f} in {run_time_s}s. "
                          F"Average speed {global_step * FLAGS.batch_size / run_time_s:.1f} records/s.")
                    sys.exit()

        epoch_stop_time = time()
        epoch_time_s = epoch_stop_time - epoch_start_time
        print(F"Finished epoch {epoch} in {datetime.timedelta(seconds=int(epoch_time_s))}. "
              F"Average speed {steps_per_epoch * FLAGS.batch_size / epoch_time_s:.1f} records/s.")

    avg_throughput = FLAGS.batch_size / metric_logger.step_time.avg

    if FLAGS.save_checkpoint_path:
        checkpoint_writer.save_checkpoint(model, FLAGS.save_checkpoint_path, epoch, step)

    results = {'best_auc': best_auc,
               'best_epoch': best_epoch,
               'average_train_throughput': avg_throughput}

    dllogger.log(data=results, step=tuple())


def dist_evaluate(model, data_loader):
    """Test distributed DLRM model

    Args:
        model (DistDLRM):
        data_loader (torch.utils.data.DataLoader):
    """
    model.eval()

    device = FLAGS.base_device
    world_size = dist.get_world_size()

    batch_sizes_per_gpu = [FLAGS.test_batch_size // world_size for _ in range(world_size)]
    test_batch_size = sum(batch_sizes_per_gpu)

    if FLAGS.test_batch_size != test_batch_size:
        print(f"Rounded test_batch_size to {test_batch_size}")
    print(f"Batch sizes per GPU {batch_sizes_per_gpu}")

    # Test bach size could be big, make sure it prints
    default_print_freq = max(524288 * 100 // test_batch_size, 1)
    print_freq = default_print_freq if FLAGS.print_freq is None else FLAGS.print_freq

    steps_per_epoch = len(data_loader)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('step_time', utils.SmoothedValue(window_size=1, fmt='{avg:.4f}'))

    with torch.no_grad():
        timer = utils.StepTimer()

        # ROC can be computed per batch and then compute AUC globally, but I don't have the code.
        # So pack all the outputs and labels together to compute AUC. y_true and y_score naming follows sklearn
        y_true = []
        y_score = []
        data_stream = torch.cuda.Stream()

        batch_iter = prefetcher(iter(data_loader), data_stream)

        timer.click()

        for step in range(len(data_loader)):
            numerical_features, categorical_features, click = next(batch_iter)
            torch.cuda.synchronize()

            last_batch_size = None
            if click.shape[0] != test_batch_size:  # last batch
                last_batch_size = click.shape[0]
                logging.warning("Pad the last test batch of size %d to %d", last_batch_size, test_batch_size)
                padding_size = test_batch_size - last_batch_size

                if numerical_features is not None:
                    padding_numerical = torch.empty(
                        padding_size, numerical_features.shape[1],
                        device=numerical_features.device, dtype=numerical_features.dtype)
                    numerical_features = torch.cat((numerical_features, padding_numerical), dim=0)

                if categorical_features is not None:
                    padding_categorical = torch.ones(
                        padding_size, categorical_features.shape[1],
                        device=categorical_features.device, dtype=categorical_features.dtype)
                    categorical_features = torch.cat((categorical_features, padding_categorical), dim=0)

            output = model(numerical_features, categorical_features, batch_sizes_per_gpu).squeeze()

            output_receive_buffer = torch.empty(test_batch_size, device=device)
            torch.distributed.all_gather(list(output_receive_buffer.split(batch_sizes_per_gpu)), output)
            if last_batch_size is not None:
                output_receive_buffer = output_receive_buffer[:last_batch_size]

            y_true.append(click)
            y_score.append(output_receive_buffer)

            timer.click()

            if timer.measured is not None:
                metric_logger.update(step_time=timer.measured)
                if step % print_freq == 0 and step > 0:
                    metric_logger.print(header=F"Test: [{step}/{steps_per_epoch}]")

        if is_main_process():
            auc = utils.roc_auc_score(torch.cat(y_true), torch.sigmoid(torch.cat(y_score).float()))
        else:
            auc = None

        torch.distributed.barrier()

    model.train()

    return auc


if __name__ == '__main__':
    app.run(main)
