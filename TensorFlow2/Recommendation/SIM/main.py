# Copyright (c) 2022 NVIDIA CORPORATION. All rights reserved.
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

import os
import random
from pathlib import Path

import click
import dllogger
import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from sim.data.dataloader import get_dataloader_tfrecord
from sim.data.defaults import FILES_SELECTOR, TEST_MAPPING, TRAIN_MAPPING
from sim.data.feature_spec import FeatureSpec
from sim.models.dien_model import DIENModel
from sim.models.din_model import DINModel
from sim.models.sim_model import SIMModel
from sim.utils.benchmark import PerformanceCalculator
from sim.utils.gpu_affinity import set_affinity
from sim.utils.losses import build_sim_loss_fn, dien_auxiliary_loss_fn
from sim.utils.misc import csv_str_to_int_list, dist_print


def init_checkpoint_manager(model, optimizer, save_checkpoint_path, load_checkpoint_path):
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=tf.Variable(-1, name='epoch')
    )

    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=save_checkpoint_path,
        max_to_keep=1,
    )

    if load_checkpoint_path != "":
        _maybe_restore_checkpoint(
            checkpoint=checkpoint,
            checkpoint_path=load_checkpoint_path
        )

    return checkpoint_manager


def _maybe_restore_checkpoint(checkpoint, checkpoint_path):
    # Needed here to support different save and load checkpoint paths
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=checkpoint_path,
        max_to_keep=1,
    )
    checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    if checkpoint_manager.latest_checkpoint:
        dist_print(f"Model restored from checkpoint {checkpoint_path}")
    else:
        dist_print(f"Failed to restore model from checkpoint {checkpoint_path}")


def init_logger(results_dir, filename):
    if hvd.rank() == 0:
        os.makedirs(results_dir, exist_ok=True)
        log_path = os.path.join(results_dir, filename)
        dllogger.init(
            backends=[
                dllogger.JSONStreamBackend(
                    verbosity=dllogger.Verbosity.VERBOSE, filename=log_path
                ),
                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE),
            ]
        )
        dllogger.metadata("test_auc", {"unit": None})
        dllogger.metadata("latency_p90", {"unit": "ms"})
        dllogger.metadata("train_loss", {"unit": None})
        dllogger.metadata("time_to_train", {"unit": "s"})
        dllogger.metadata("throughput", {"unit": "samples/s"})
    else:
        dllogger.init(backends=[])


# In the future, select one of available dataloaders there (tfrecord, csv, etc...)
def get_data_iterator(paths, feature_spec, batch_size, num_gpus, long_seq_length, prefetch_size, num_parallel_calls=None, repeat_count=0,
                      drop_remainder=False, amp=False, disable_cache=False, prebatch_size=0):
    return get_dataloader_tfrecord(
        paths,
        feature_spec=feature_spec,
        batch_size=batch_size,
        long_seq_length=long_seq_length,
        num_gpus=num_gpus,
        id=hvd.rank(),
        drop_remainder=drop_remainder,
        repeat_count=repeat_count,
        disable_cache=disable_cache,
        prefetch_buffer_size=prefetch_size,
        num_parallel_calls=num_parallel_calls,
        prebatch_size=prebatch_size
    )


def build_model_and_loss(model_params):
    model_type = model_params["model_type"]

    if model_type == "sim":
        model = SIMModel(
            model_params['feature_spec'],
            mlp_hidden_dims=model_params["mlp_hidden_dims"],
            embedding_dim=model_params["embedding_dim"],
            dropout_rate=model_params["dropout_rate"]
        )
        classification_loss_fn = build_sim_loss_fn()

        @tf.function
        def model_fn(batch, training=True):
            input_data, targets = batch
            # take the mask for N-1 timesteps from prepared input data
            mask_for_aux_loss = input_data["short_sequence_mask"][:, 1:]

            # model forward pass
            output_dict = model(input_data, training=training)

            # compute loss
            classification_loss = classification_loss_fn(
                targets, output_dict["stage_one_logits"], output_dict["stage_two_logits"]
            )

            dien_aux_loss = dien_auxiliary_loss_fn(
                output_dict["aux_click_probs"],
                output_dict["aux_noclick_probs"],
                mask=mask_for_aux_loss,
            )

            total_loss = classification_loss + dien_aux_loss

            logits = output_dict["stage_two_logits"]

            loss_dict = {
                "total_loss": total_loss,
                "classification_loss": classification_loss,
                "dien_aux_loss": dien_aux_loss
            }

            return (targets, logits), loss_dict
    elif model_type == "dien":
        model = DIENModel(
            model_params['feature_spec'],
            mlp_hidden_dims={
                "classifier": model_params["mlp_hidden_dims"]["stage_2"],
                "aux": model_params["mlp_hidden_dims"]["aux"],
            },
            embedding_dim=model_params["embedding_dim"],
        )
        classification_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        @tf.function
        def model_fn(batch, training=True):
            input_data, targets = batch
            # take the mask for N-1 timesteps from prepared input data
            mask_for_aux_loss = input_data["short_sequence_mask"][:, 1:]

            # model forward pass
            output_dict = model(input_data, training=training)

            # compute loss
            classification_loss = classification_loss_fn(targets, output_dict["logits"])

            dien_aux_loss = dien_auxiliary_loss_fn(
                output_dict["aux_click_probs"],
                output_dict["aux_noclick_probs"],
                mask=mask_for_aux_loss,
            )

            total_loss = classification_loss + dien_aux_loss

            logits = output_dict["logits"]

            loss_dict = {
                "total_loss": total_loss,
                "classification_loss": classification_loss,
                "dien_aux_loss": dien_aux_loss
            }

            return (targets, logits), loss_dict
    elif model_type == "din":
        model = DINModel(
            model_params['feature_spec'],
            mlp_hidden_dims=model_params["mlp_hidden_dims"]["stage_2"],
            embedding_dim=model_params["embedding_dim"]
        )
        classification_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        @tf.function
        def model_fn(batch, training=True):
            input_data, targets = batch

            # model forward pass
            output_dict = model(input_data, training=training)

            # compute loss
            total_loss = classification_loss_fn(
                targets, output_dict["logits"]
            )

            logits = output_dict["logits"]

            loss_dict = {"total_loss": total_loss}

            return (targets, logits), loss_dict

    return model, model_fn


@tf.function
def _update_auc(auc_accumulator, targets, logits):
    auc_accumulator.update_state(targets, logits)


def eval(model_fn, data_iterator, num_thresholds=8000, prefix=""):
    auc_accumulator = tf.keras.metrics.AUC(
        num_thresholds=num_thresholds, name="auc_accumulator", from_logits=True
    )

    distributed = hvd.size() != 1

    local_logits = []
    local_targets = []
    local_total_losses = []

    for batch in data_iterator:
        (targets, logits), loss_dict = model_fn(batch, training=False)
        local_logits.append(logits)
        local_targets.append(targets)
        local_total_losses.append(loss_dict["total_loss"])

    locals = [local_logits, local_targets, local_total_losses]
    for i, local in enumerate(locals):

        # wrap empty lists in tensor to allow tf.concat
        if len(local) == 0:
            local = tf.constant(local)

        # concat all local variables into a single tensor
        if local is local_total_losses:
            local = tf.stack(local, 0)
        else:
            local = tf.concat(local, 0)

        # for single element lists, tf.concat will produce shape=() instead of shape=(1,).
        # reshape it for hvd.allgather to work
        if len(local.shape) == 0:
            local = tf.reshape(local, -1)

        locals[i] = local
    
    logits, targets, total_losses = locals

    if distributed:
        # gather from all nodes
        logits = hvd.allgather(logits)
        targets = hvd.allgather(targets)
        total_losses = hvd.allgather(total_losses)

    if hvd.rank() == 0:
        # need to convert it to a dataset first
        split_batch_size = local_logits[0].shape[0]
        metrics_ds = tf.data.Dataset.from_tensor_slices((targets, logits)).batch(split_batch_size)
        # run batched version of metrics update
        for targets, logits in metrics_ds:
            _update_auc(auc_accumulator, targets, logits)
        loss = tf.reduce_mean(total_losses).numpy().item()
        auc = auc_accumulator.result().numpy().item()
    else:
        loss = 0.
        auc = 0.
    return {f"{prefix}auc": auc, f"{prefix}loss": loss}


@tf.function
def model_step(batch, model, model_fn, optimizer, amp, first_batch):
    with tf.GradientTape() as tape:
        _, loss_dict = model_fn(batch, training=True)
        loss = loss_dict["total_loss"]
        scaled_loss = optimizer.get_scaled_loss(loss) if amp else loss

    tape = hvd.DistributedGradientTape(tape, sparse_as_dense=True, compression=hvd.Compression.fp16)
    grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = optimizer.get_unscaled_gradients(grads) if amp else grads

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(optimizer.variables(), root_rank=0)

    return loss_dict


def run_single_epoch(model, model_fn, data_iterator, optimizer, amp, start_epoch, epoch, benchmark, performance_calculator):

    for current_step, batch in enumerate(data_iterator):
        if benchmark and performance_calculator.completed:
            break

        is_first_batch = (current_step == 0 and epoch == 0)
        step_dict = model_step(batch, model, model_fn, optimizer, amp, is_first_batch)
        step_dict = {key: val.numpy().item() for key, val in step_dict.items()}

        n_samples = len(batch[1])
        step_throughput = performance_calculator(n_samples)
        step_dict["samples/s"] = step_throughput
        dllogger.log(data=step_dict, step=(start_epoch + epoch, current_step))


def train(model, model_fn, data_iterator_train, data_iterator_test, optimizer, amp, epochs,
          benchmark, performance_calculator, save_checkpoint, checkpoint_manager):
    """Train and evaluate the model for a given number of epochs."""

    performance_calculator.init()

    all_epochs_results = []

    start_epoch = checkpoint_manager.checkpoint.epoch.numpy().item() + 1

    for epoch in range(epochs - start_epoch):
        run_single_epoch(model, model_fn, data_iterator_train, optimizer, amp, start_epoch, epoch, benchmark, performance_calculator)

        if not benchmark:
            # we dump throughput results for consecutive epochs for a regular training job (w/o --benchmark flag)
            results_data = performance_calculator.get_current_benchmark_results()
            all_epochs_results.append(results_data)

            results_eval_train = eval(model_fn, data_iterator_train, prefix="train_")
            results_eval_test = eval(model_fn, data_iterator_test, prefix="test_")
            results_data.update(results_eval_train)
            results_data.update(results_eval_test)

            if save_checkpoint:
                checkpoint_manager.checkpoint.epoch.assign(epoch)
                checkpoint_manager.save()

            if hvd.rank() == 0:
                dllogger.log(data=results_data, step=(start_epoch + epoch,))

            performance_calculator.init()  # restart for another epoch

        elif performance_calculator.completed:
            break

    if benchmark:
        results_perf = performance_calculator.results
        if not performance_calculator.completed:
            # model steps have been exhausted or all steps should be included to calculate throughput
            results_perf = performance_calculator.get_current_benchmark_results()

        if hvd.rank() == 0:
            dllogger.log(data=results_perf, step=tuple())
    else:
        # calculate convergence metrics
        time_to_train = sum([epoch_result['time'] for epoch_result in all_epochs_results])
        results = {'time_to_train': time_to_train}
        results.update(results_eval_train)
        results.update(results_eval_test)

        if hvd.rank() == 0:
            dllogger.log(data=results, step=tuple())


def inference(model, data_iterator, benchmark, performance_calculator):
    """Forward pass for the model and data loader given."""
    performance_calculator.init()

    for current_step, (input_data, targets) in enumerate(data_iterator):
        if benchmark and performance_calculator.completed:
            break
        model(input_data, training=False, compute_aux_loss=False)
        step_throughput = performance_calculator(len(targets))
        dllogger.log(data={"samples/s": step_throughput}, step=(0, current_step))

    results_perf = performance_calculator.results
    if not performance_calculator.completed:
        results_perf = performance_calculator.get_current_benchmark_results()

    if hvd.rank() == 0:
        dllogger.log(data=results_perf, step=tuple())


@click.command()
@click.option(
    "--mode",
    default="train",
    help="Script mode: available options are 'train' to train and evaluate the model "
         "and 'inference' to perform forward pass over a given dataset",
    type=click.Choice(["train", "inference"]),
)
@click.option(
    "--dataset_dir",
    required=True,
    help="Path to the dataset directory.",
    type=str,
)
@click.option(
    "--feature_spec",
    default='feature_spec.yaml',
    help="Name of the feature spec file in the dataset directory.",
    type=str
)
@click.option(
    "--results_dir",
    default="/tmp/sim",
    help="Path to the model files storage.",
    type=str,
)
@click.option(
    "--log_filename",
    default="log.json",
    help="Name of the file to store dllogger output.",
    type=str,
)
@click.option(
    "--long_seq_length",
    default=90,
    help="length of long history sequence",
    type=int
)
@click.option(
    "--optimizer",
    default="adam",
    help="Optimizer to use [adam/lazy_adam/sgd].",
    type=click.Choice(["adam", "lazy_adam", "sgd"]),
)
@click.option(
    "--affinity",
    default="socket_unique_interleaved",
    help="Type of CPU affinity",
    type=click.Choice([
        "socket",
        "single",
        "single_unique",
        "socket_unique_interleaved",
        "socket_unique_continuous",
        "disabled",
    ],
    ),
)
@click.option(
    "--seed", default=-1, help="Random seed.", type=int
)
@click.option(
    "--lr", default=0.01, help="Learning rate of the selected optimizer.", type=float
)
@click.option(
    "--dropout_rate", default=-1, help="Dropout rate for all the classification MLPs (default: -1, disabled).",
    type=float
)
@click.option(
    "--weight_decay", default=0, help="Parameters decay of the selected optimizer.", type=float
)
@click.option(
    "--embedding_dim", default=16, help="Embedding dimension.", type=int
)
@click.option(
    "--global_batch_size", default=131072, help="Batch size used to train/eval the model.", type=int
)
@click.option(
    "--num_parallel_calls", default=None, help="Parallelism level for tf.data API. If None, heuristic based on number of CPUs and number of GPUs will be used."
)
@click.option(
    "--epochs", default=3, help="Train for the following number of epochs.", type=int
)
@click.option("--disable_cache", help="disable dataset caching.", is_flag=True)
@click.option("--drop_remainder", help="Drop remainder batch for training set.", is_flag=True)
@click.option(
    "--repeat_count", default=0, help="Repeat training dataset this number of times.", type=int
)
@click.option(
    "--benchmark",
    is_flag=True
)
@click.option(
    "--benchmark_steps",
    default=0,
    help="Number of steps to use for performance benchmarking. Use benchmark_steps <= 0 to include all iterations. "
         "This parameter has no effect when the script is launched without --benchmark flag.",
    type=int
)
@click.option(
    "--benchmark_warmup_steps",
    default=20,
    help="Number of warmup steps to use for performance benchmarking (benchmark_warmup_steps <= 0 means no warmup).",
    type=int
)
@click.option(
    "--stage_one_mlp_dims",
    default="200",
    help="MLP hidden dimensions for stage one (excluding classification output).",
    type=str,
)
@click.option(
    "--stage_two_mlp_dims",
    default="200,80",
    help="MLP hidden dimensions for stage two (excluding classification output).",
    type=str,
)
@click.option(
    "--aux_mlp_dims",
    default="100,50",
    help="MLP hidden dimensions for aux loss (excluding classification output).",
    type=str,
)
@click.option(
    "--model_type",
    default="sim",
    type=click.Choice(["sim", "din", "dien"])
)
@click.option("--save_checkpoint_path", default="", type=str)
@click.option("--load_checkpoint_path", default="", type=str)
@click.option("--amp", is_flag=True)
@click.option("--xla", is_flag=True)
@click.option(
    "--inter_op_parallelism",
    default=0,
    help="Number of inter op threads.",
    type=int
)
@click.option(
    "--intra_op_parallelism",
    default=0,
    help="Number of intra op threads.",
    type=int
)
@click.option(
    "--prefetch_train_size",
    default=10,
    help="Number of batches to prefetch in training. "
)
@click.option(
    "--prefetch_test_size",
    default=2,
    help="Number of batches to prefetch in testing"
)
@click.option(
    "--prebatch_train_size",
    default=0,
    help="Information about batch size applied during preprocessing to train dataset"
)
@click.option(
    "--prebatch_test_size",
    default=0,
    help="Information about batch size applied during preprocessing to test dataset"
)
def main(
        mode: str,
        dataset_dir: str,
        feature_spec: str,
        results_dir: str,
        log_filename: str,
        long_seq_length: int,
        save_checkpoint_path: str,
        load_checkpoint_path: str,
        model_type: str,
        optimizer: str,
        affinity: str,
        seed: int,
        lr: float,
        dropout_rate: float,
        weight_decay: float,
        embedding_dim: int,
        global_batch_size: int,
        num_parallel_calls: int,
        epochs: int,
        disable_cache: bool,
        drop_remainder: bool,
        repeat_count: int,
        benchmark: bool,
        benchmark_steps: int,
        benchmark_warmup_steps: int,
        stage_one_mlp_dims: str,
        stage_two_mlp_dims: str,
        aux_mlp_dims: str,
        xla: bool,
        amp: bool,
        inter_op_parallelism: int,
        intra_op_parallelism: int,
        prefetch_train_size: int,
        prefetch_test_size: int,
        prebatch_train_size: int,
        prebatch_test_size: int
):
    hvd.init()

    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    if affinity != "disabled":
        gpu_id = hvd.local_rank()
        affinity = set_affinity(
            gpu_id=gpu_id, nproc_per_node=hvd.size(), mode=affinity
        )
        dist_print(f"{gpu_id}: thread affinity: {affinity}")

    init_logger(results_dir, log_filename)

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    if amp:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    if inter_op_parallelism > 0:
        tf.config.threading.set_inter_op_parallelism_threads(inter_op_parallelism)

    if intra_op_parallelism > 0:
        tf.config.threading.set_intra_op_parallelism_threads(intra_op_parallelism)

    if xla:
        tf.config.optimizer.set_jit(True)

    if optimizer == "adam":
        optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    elif optimizer == "lazy_adam":
        optimizer = tfa.optimizers.LazyAdam(lr)
    elif optimizer == "sgd":
        optimizer = tfa.optimizers.SGDW(learning_rate=lr, weight_decay=weight_decay)

    optimizer = hvd.DistributedOptimizer(optimizer,  compression=hvd.Compression.fp16)
    if amp:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)

    num_gpus = hvd.size()
    if global_batch_size % num_gpus != 0:
        raise ValueError('Global batch size must be divisible by number of gpus. Otherwise it may result in deadlock.')

    batch_size = global_batch_size // num_gpus

    """
    In case of:
        - benchmark: we can load only 1 batch and operate on it for benchmark_steps times (in preload fashion).
        - training: we can repeat via a flag
    """

    dataset_dir = Path(dataset_dir)

    feature_spec = FeatureSpec.from_yaml(dataset_dir / feature_spec)

    # since each tfrecord file must include all of the features, it is enough to read first chunk for each split. 
    train_files = [dataset_dir / file for file in feature_spec.source_spec[TRAIN_MAPPING][0][FILES_SELECTOR]]

    data_iterator_train = get_data_iterator(
        train_files, feature_spec, batch_size, num_gpus, long_seq_length,
        repeat_count=repeat_count, drop_remainder=drop_remainder,
        amp=amp, disable_cache=disable_cache, prefetch_size=prefetch_train_size,
        num_parallel_calls=num_parallel_calls, prebatch_size=prebatch_train_size
    )

    if mode == "train":
        test_files = [dataset_dir / file for file in feature_spec.source_spec[TEST_MAPPING][0][FILES_SELECTOR]]
        data_iterator_test = get_data_iterator(
            test_files, feature_spec, batch_size, num_gpus, long_seq_length,
            amp=amp, disable_cache=disable_cache, prefetch_size=prefetch_test_size, num_parallel_calls=num_parallel_calls,
            prebatch_size=prebatch_test_size
        )
    else:
        data_iterator_test = []  # otherwise not used

    stage_one_mlp_dims = csv_str_to_int_list(stage_one_mlp_dims)
    stage_two_mlp_dims = csv_str_to_int_list(stage_two_mlp_dims)
    aux_mlp_dims = csv_str_to_int_list(aux_mlp_dims)

    model_params = {
        "feature_spec": feature_spec,
        "embedding_dim": embedding_dim,
        "mlp_hidden_dims": {
            "stage_1": stage_one_mlp_dims,
            "stage_2": stage_two_mlp_dims,
            "aux": aux_mlp_dims
        },
        "dropout_rate": dropout_rate,
        "model_type": model_type
    }

    model, model_fn = build_model_and_loss(model_params)
    checkpoint_manager = init_checkpoint_manager(
        model, optimizer,
        save_checkpoint_path, load_checkpoint_path
    )
    save_checkpoint = save_checkpoint_path != "" and hvd.rank() == 0

    performance_calculator = PerformanceCalculator(
        benchmark_warmup_steps, benchmark_steps
    )

    if mode == "train":
        train(model, model_fn, data_iterator_train, data_iterator_test, optimizer, amp, epochs,
              benchmark, performance_calculator, save_checkpoint, checkpoint_manager)
    elif mode == "inference":
        inference(model, data_iterator_train, benchmark, performance_calculator)


if __name__ == "__main__":
    main()
