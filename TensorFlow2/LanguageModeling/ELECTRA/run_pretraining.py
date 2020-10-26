# coding=utf-8
# Copyright 2020 The Google Research Authors.
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pre-trains an ELECTRA model."""

import argparse
import collections
import json
import time
import datetime
import os

import tensorflow as tf
import horovod.tensorflow as hvd
from horovod.tensorflow.compression import Compression
from gpu_affinity import set_affinity

import utils
import sys
import pretrain_utils
from utils import get_rank, get_world_size, is_main_process, log, log_config, setup_logger, postprocess_dllog
from tokenization import ElectraTokenizer
from modeling import PretrainingModel
from optimization import create_optimizer, GradientAccumulator
import dllogger

class PretrainingConfig(object):
    """Defines pre-training hyperparameters."""

    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.seed = 42

        self.debug = False  # debug mode for quickly running things
        self.do_train = True  # pre-train ELECTRA
        self.do_eval = False  # evaluate generator/discriminator on unlabeled data
        self.phase2 = False

        # amp
        self.amp = True
        self.xla = True
        self.fp16_compression = False

        # optimizer type
        self.optimizer = 'adam'
        self.gradient_accumulation_steps = 1

        # lamb whitelisting for LN and biases
        self.skip_adaptive = False

        # loss functions
        self.electra_objective = True  # if False, use the BERT objective instead
        self.gen_weight = 1.0  # masked language modeling / generator loss
        self.disc_weight = 50.0  # discriminator loss
        self.mask_prob = 0.15  # percent of input tokens to mask out / replace

        # optimization
        self.learning_rate = 5e-4
        self.lr_decay_power = 0.5
        self.weight_decay_rate = 0.01
        self.num_warmup_steps = 10000
        self.opt_beta_1 = 0.878
        self.opt_beta_2 = 0.974
        self.end_lr = 0.0

        # training settings
        self.log_freq = 10
        self.skip_checkpoint = False
        self.save_checkpoints_steps = 1000
        self.num_train_steps = 1000000
        self.num_eval_steps = 100
        self.keep_checkpoint_max = 5  # maximum number of recent checkpoint files to keep;  change to 0 or None to keep all checkpoints
        self.restore_checkpoint = None
        self.load_weights = False

        # model settings
        self.model_size = "base"  # one of "small", "base", or "large"
        # override the default transformer hparams for the provided model size; see
        # modeling.BertConfig for the possible hparams and util.training_utils for
        # the defaults
        self.model_hparam_overrides = (
            kwargs["model_hparam_overrides"]
            if "model_hparam_overrides" in kwargs else {})
        self.embedding_size = None  # bert hidden size by default
        self.vocab_size = 30522  # number of tokens in the vocabulary
        self.do_lower_case = True  # lowercase the input?

        # generator settings
        self.uniform_generator = False  # generator is uniform at random
        self.shared_embeddings = True  # share generator/discriminator token embeddings?
        # self.untied_generator = True  # tie all generator/discriminator weights?
        self.generator_layers = 1.0  # frac of discriminator layers for generator
        self.generator_hidden_size = 0.25  # frac of discrim hidden size for gen
        self.disallow_correct = False  # force the generator to sample incorrect
        # tokens (so 15% of tokens are always
        # fake)
        self.temperature = 1.0  # temperature for sampling from generator

        # batch sizes
        self.max_seq_length = 128
        self.train_batch_size = 128
        self.eval_batch_size = 128

        self.results_dir = "results"
        self.json_summary = None
        self.update(kwargs)
        # default locations of data files
        
        self.pretrain_tfrecords = os.path.join(
            "data", "pretrain_tfrecords/pretrain_data.tfrecord*")
        self.vocab_file = os.path.join("vocab", "vocab.txt")
        self.model_dir = os.path.join(self.results_dir, "models", model_name)
        self.checkpoints_dir = os.path.join(self.model_dir, "checkpoints")
        self.weights_dir = os.path.join(self.model_dir, "weights")
        self.results_txt = os.path.join(self.results_dir, "unsup_results.txt")
        self.results_pkl = os.path.join(self.results_dir, "unsup_results.pkl")
        self.log_dir = os.path.join(self.model_dir, "logs")

        self.max_predictions_per_seq = int((self.mask_prob + 0.005) *
                                           self.max_seq_length)

        # defaults for different-sized model
        if self.model_size == "base":
            self.embedding_size = 768
            self.hidden_size = 768
            self.num_hidden_layers = 12
            if self.hidden_size % 64 != 0:
                raise ValueError("Hidden size {} should be divisible by 64. Number of attention heads is hidden size {} / 64 ".format(self.hidden_size, self.hidden_size))	
            self.num_attention_heads = int(self.hidden_size / 64.)
        elif self.model_size == "large":
            self.embedding_size = 1024
            self.hidden_size = 1024
            self.num_hidden_layers = 24
            if self.hidden_size % 64 != 0:
                raise ValueError("Hidden size {} should be divisible by 64. Number of attention heads is hidden size {} / 64 ".format(self.hidden_size, self.hidden_size))
            self.num_attention_heads = int(self.hidden_size / 64.)
        else:
            raise ValueError("--model_size : 'base' and 'large supported only.")
        self.act_func = "gelu"
        self.hidden_dropout_prob = 0.1 
        self.attention_probs_dropout_prob = 0.1

        self.update(kwargs)

    def update(self, kwargs):
        for k, v in kwargs.items():
            if v is not None:
                self.__dict__[k] = v


def metric_fn(config, metrics, eval_fn_inputs):
    """Computes the loss and accuracy of the model."""
    d = eval_fn_inputs
    metrics["masked_lm_accuracy"].update_state(
        y_true=tf.reshape(d["masked_lm_ids"], [-1]),
        y_pred=tf.reshape(d["masked_lm_preds"], [-1]),
        sample_weight=tf.reshape(d["masked_lm_weights"], [-1]))
    metrics["masked_lm_loss"].update_state(
        values=tf.reshape(d["mlm_loss"], [-1]),
        sample_weight=tf.reshape(d["masked_lm_weights"], [-1]))
    if config.electra_objective:
        metrics["sampled_masked_lm_accuracy"].update_state(
            y_true=tf.reshape(d["masked_lm_ids"], [-1]),
            y_pred=tf.reshape(d["sampled_tokids"], [-1]),
            sample_weight=tf.reshape(d["masked_lm_weights"], [-1]))
        if config.disc_weight > 0:
            metrics["disc_loss"].update_state(d["disc_loss"])
            #metrics["disc_auc"].update_state(
            #    d["disc_labels"] * d["input_mask"],
            #    d["disc_probs"] * tf.cast(d["input_mask"], tf.float32))
            metrics["disc_accuracy"].update_state(
                y_true=d["disc_labels"], y_pred=d["disc_preds"],
                sample_weight=d["input_mask"])
            metrics["disc_precision"].update_state(
                y_true=d["disc_labels"], y_pred=d["disc_preds"],
                sample_weight=d["disc_preds"] * d["input_mask"])
            metrics["disc_recall"].update_state(
                y_true=d["disc_labels"], y_pred=d["disc_preds"],
                sample_weight=d["disc_labels"] * d["input_mask"])
    return metrics

@tf.function
def train_one_step(config, model, optimizer, features, accumulator, first_step, take_step, clip_norm=1.0):

    #Forward and Backward pass
    with tf.GradientTape() as tape:
        total_loss, eval_fn_inputs = model(features, is_training=True)
        unscaled_loss = tf.stop_gradient(total_loss)
        if config.amp:
            total_loss = optimizer.get_scaled_loss(total_loss)
   
    #Backpropogate gradients
    #tape = hvd.DistributedGradientTape(
    #    tape, sparse_as_dense=True,
    #    compression=Compression.fp16 if config.amp and config.fp16_compression else Compression.none)
    gradients = tape.gradient(total_loss, model.trainable_variables)

    #Get unscaled gradients if AMP
    if config.amp:
        gradients = optimizer.get_unscaled_gradients(gradients)

    #Accumulate gradients
    accumulator(gradients)
    #Need to call apply_gradients on very first step irrespective of gradient accumulation
    #This is required for the optimizer to build it's states
    if first_step or take_step:
        #All reduce and Clip the accumulated gradients
        allreduced_accumulated_gradients = [None if g is None else hvd.allreduce(g / tf.cast(config.gradient_accumulation_steps, g.dtype),
                                compression=Compression.fp16 if config.amp and config.fp16_compression else Compression.none)
                                for g in accumulator.gradients]
        (clipped_accumulated_gradients, _) = tf.clip_by_global_norm(allreduced_accumulated_gradients, clip_norm=clip_norm)
        #Weight update
        optimizer.apply_gradients(zip(clipped_accumulated_gradients, model.trainable_variables))
        accumulator.reset()

    #brodcast model weights after first train step
    if first_step:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(optimizer.variables(), root_rank=0)

    return unscaled_loss, eval_fn_inputs

def main(e2e_start_time):
    # Parse essential argumentss
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_size", default="base", type=str, help="base or large")
    parser.add_argument("--pretrain_tfrecords", type=str)
    parser.add_argument("--phase2", action='store_true')
    parser.add_argument("--fp16_compression", action='store_true')
    parser.add_argument("--amp", action='store_true',
                        help="Whether to use fp16.")
    parser.add_argument("--xla", action='store_true',
                        help="Whether to use xla.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_train_steps", type=int)
    parser.add_argument("--num_warmup_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--max_seq_length", type=int)

    parser.add_argument("--mask_prob", type=float)
    parser.add_argument("--disc_weight", type=float)
    parser.add_argument("--generator_hidden_size", type=float)

    parser.add_argument("--log_freq", type=int, default=10, help="Training metrics logging frequency")
    parser.add_argument("--save_checkpoints_steps", type=int)
    parser.add_argument("--keep_checkpoint_max", type=int)
    parser.add_argument("--restore_checkpoint", default=None, type=str)
    parser.add_argument("--load_weights", action='store_true')
    parser.add_argument("--weights_dir")

    parser.add_argument("--optimizer", default="adam", type=str, help="adam or lamb")
    parser.add_argument("--skip_adaptive", action='store_true', help="Whether to apply adaptive LR on LayerNorm and biases")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of Gradient Accumulation steps")
    parser.add_argument("--lr_decay_power", type=float, default=0.5, help="LR decay power")
    parser.add_argument("--opt_beta_1", type=float, default=0.878, help="Optimizer beta1")
    parser.add_argument("--opt_beta_2", type=float, default=0.974, help="Optimizer beta2")
    parser.add_argument("--end_lr", type=float, default=0.0, help="Ending LR")
    parser.add_argument("--log_dir", type=str, default=None, help="Path to store logs")
    parser.add_argument("--results_dir", type=str, default=None, help="Path to store all model results")
    parser.add_argument("--skip_checkpoint", action='store_true', default=False, help="Path to store logs")
    parser.add_argument('--json-summary', type=str, default=None,
                        help='If provided, the json summary will be written to the specified file.')
    args = parser.parse_args()
    config = PretrainingConfig(**args.__dict__)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    # Set up tensorflow
    hvd.init()

    args.log_dir = config.log_dir
    # DLLogger
    setup_logger(args)

    set_affinity(hvd.local_rank())
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tf.config.optimizer.set_jit(config.xla)
    #tf.config.optimizer.set_experimental_options({"auto_mixed_precision": config.amp})

    if config.amp:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16", loss_scale="dynamic")
        tf.keras.mixed_precision.experimental.set_policy(policy)
        print('Compute dtype: %s' % policy.compute_dtype)  # Compute dtype: float16
        print('Variable dtype: %s' % policy.variable_dtype)  # Variable dtype: float32

    #tf.random.set_seed(config.seed)

    # Set up config cont'
    if config.load_weights and config.restore_checkpoint:
        raise ValueError("`load_weights` and `restore_checkpoint` should not be on at the same time.")
    if config.phase2 and not config.restore_checkpoint:
        raise ValueError("`phase2` cannot be used without `restore_checkpoint`.")
    utils.heading("Config:")
    log_config(config)

    # Save pretrain configs
    pretrain_config_json = os.path.join(config.checkpoints_dir, 'pretrain_config.json')
    if is_main_process():
        utils.write_json(config.__dict__, pretrain_config_json)
        log("Configuration saved in {}".format(pretrain_config_json))

    # Set up model
    model = PretrainingModel(config)

    # Set up metrics
    metrics = dict()
    metrics["train_perf"] = tf.keras.metrics.Mean(name="train_perf")
    metrics["total_loss"] = tf.keras.metrics.Mean(name="total_loss")
    metrics["masked_lm_accuracy"] = tf.keras.metrics.Accuracy(name="masked_lm_accuracy")
    metrics["masked_lm_loss"] = tf.keras.metrics.Mean(name="masked_lm_loss")
    if config.electra_objective:
        metrics["sampled_masked_lm_accuracy"] = tf.keras.metrics.Accuracy(name="sampled_masked_lm_accuracy")
        if config.disc_weight > 0:
            metrics["disc_loss"] = tf.keras.metrics.Mean(name="disc_loss")
            metrics["disc_auc"] = tf.keras.metrics.AUC(name="disc_auc")
            metrics["disc_accuracy"] = tf.keras.metrics.Accuracy(name="disc_accuracy")
            metrics["disc_precision"] = tf.keras.metrics.Accuracy(name="disc_precision")
            metrics["disc_recall"] = tf.keras.metrics.Accuracy(name="disc_recall")

    # Set up tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(config.log_dir, current_time,
                                 'train_' + str(get_rank()) + '_of_' + str(get_world_size()))
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Set up dataset
    dataset = pretrain_utils.get_dataset(
        config, config.train_batch_size, world_size=get_world_size(), rank=get_rank())
    train_iterator = iter(dataset)

    # Set up optimizer
    optimizer = create_optimizer(
        init_lr=config.learning_rate,
        num_train_steps=config.num_train_steps,
        num_warmup_steps=config.num_warmup_steps,
        weight_decay_rate=config.weight_decay_rate,
        optimizer=config.optimizer,
        skip_adaptive=config.skip_adaptive,
        power=config.lr_decay_power,
        beta_1=config.opt_beta_1,
        beta_2=config.opt_beta_2,
        end_lr=config.end_lr)
        
    accumulator = GradientAccumulator()
    if config.amp:
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

    # Set up model checkpoint
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0), phase2=tf.Variable(False), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, config.checkpoints_dir, max_to_keep=config.keep_checkpoint_max)
    if config.restore_checkpoint and config.restore_checkpoint != "latest":
        checkpoint.restore(config.restore_checkpoint)
        log(" ** Restored model checkpoint from {}".format(config.restore_checkpoint))
    elif config.restore_checkpoint and config.restore_checkpoint == "latest" and manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        log(" ** Restored model checkpoint from {}".format(manager.latest_checkpoint))
    elif config.load_weights:
        model.generator(model.generator.dummy_inputs)
        model.discriminator(model.discriminator.dummy_inputs)
        model.generator.load_weights(os.path.join(config.weights_dir, 'generator', 'tf_model.h5'))
        model.discriminator.load_weights(os.path.join(config.weights_dir, 'discriminator', 'tf_model.h5'))
    else:
        log(" ** Initializing from scratch.")

    restore_iterator = bool(config.restore_checkpoint) and config.restore_checkpoint == "latest"
    # Initialize global step for phase2
    if config.phase2 and not bool(checkpoint.phase2):
        optimizer.iterations.assign(0)
        checkpoint.step.assign(0)
        checkpoint.phase2.assign(True)
        restore_iterator = False
    if bool(checkpoint.phase2):
        manager = tf.train.CheckpointManager(
            checkpoint, config.checkpoints_dir,
            checkpoint_name='ckpt-p2',
            max_to_keep=config.keep_checkpoint_max)

    # Set up iterator checkpoint
    iter_checkpoint = tf.train.Checkpoint(
        train_iterator=train_iterator, world_size=tf.Variable(get_world_size()), rank=tf.Variable(get_rank()))
    iter_manager = tf.train.CheckpointManager(
        iter_checkpoint,
        os.path.join(config.checkpoints_dir, 'iter_ckpt_rank_' + '{:02}'.format(get_rank())),
        checkpoint_name='iter_ckpt_rank_' + '{:02}'.format(get_rank()),
        max_to_keep=config.keep_checkpoint_max)
    if restore_iterator and iter_manager.latest_checkpoint:
        ckpt_world_size = tf.train.load_variable(
            iter_manager.latest_checkpoint, 'world_size/.ATTRIBUTES/VARIABLE_VALUE')
        if ckpt_world_size == get_world_size():
            iter_checkpoint.restore(iter_manager.latest_checkpoint)
            log(" ** Restored iterator checkpoint from {}".format(iter_manager.latest_checkpoint), all_rank=True)

    utils.heading("Running training")
    accumulator.reset()
    train_start, start_step = time.time(), int(checkpoint.step) - 1
    local_step = 0
    saved_ckpt = False
    while int(checkpoint.step) <= config.num_train_steps:
        saved_ckpt = False
        step = int(checkpoint.step)
        features = next(train_iterator)
        iter_start = time.time()

        # if step == 200: tf.profiler.experimental.start(logdir=train_log_dir)
        total_loss, eval_fn_inputs = train_one_step(config, model, optimizer, features, accumulator,
                                                    local_step==1, take_step=local_step % args.gradient_accumulation_steps == 0)
        # if step == 300: tf.profiler.experimental.stop()

        metrics["train_perf"].update_state(
            config.train_batch_size * get_world_size() / (time.time() - iter_start))
        metrics["total_loss"].update_state(values=total_loss)
        metric_fn(config, metrics, eval_fn_inputs)

        if (step % args.log_freq == 0) and (local_step % args.gradient_accumulation_steps == 0):
            log_info_dict = {k:float(v.result().numpy() * 100) if "accuracy" in k else float(v.result().numpy()) for k, v in metrics.items()}
            dllogger.log(step=(step,), data=log_info_dict, verbosity=0)
            log('Step:{step:6d}, Loss:{total_loss:10.6f}, Gen_loss:{masked_lm_loss:10.6f}, Disc_loss:{disc_loss:10.6f}, Gen_acc:{masked_lm_accuracy:6.2f}, '
                'Disc_acc:{disc_accuracy:6.2f}, Perf:{train_perf:4.0f}, Loss Scaler: {loss_scale}, Elapsed: {elapsed}, ETA: {eta}, '.format(
                step=step, **log_info_dict,
                loss_scale=optimizer.loss_scale if config.amp else 1,
                elapsed=utils.get_readable_time(time.time() - train_start),
                eta=utils.get_readable_time(
                    (time.time() - train_start) / (step - start_step) * (config.num_train_steps - step))),
                all_rank=True)

            with train_summary_writer.as_default():
                for key, m in metrics.items():
                    tf.summary.scalar(key, m.result(), step=step)

            if int(checkpoint.step) < config.num_train_steps:
                for m in metrics.values():
                    m.reset_states()

        #Print allreduced metrics on the last step
        if int(checkpoint.step) == config.num_train_steps and (local_step % args.gradient_accumulation_steps == 0):
            log_info_dict = {k:float(hvd.allreduce(v.result()).numpy() * 100) if "accuracy" in k else float(hvd.allreduce(v.result()).numpy()) for k, v in metrics.items()}
            log_info_dict["training_sequences_per_second"] = log_info_dict["train_perf"]
            log_info_dict["final_loss"] = log_info_dict["total_loss"]
            log_info_dict["e2e_train_time"] = time.time() - e2e_start_time
            dllogger.log(step=(), data=log_info_dict, verbosity=0)
            log('<FINAL STEP METRICS> Step:{step:6d}, Loss:{total_loss:10.6f}, Gen_loss:{masked_lm_loss:10.6f}, Disc_loss:{disc_loss:10.6f}, Gen_acc:{masked_lm_accuracy:6.2f}, '
                'Disc_acc:{disc_accuracy:6.2f}, Perf:{train_perf:4.0f},'.format(
                step=step, **log_info_dict),
                all_rank=False)

        if local_step % args.gradient_accumulation_steps == 0:
            checkpoint.step.assign(int(optimizer.iterations))
        
        local_step += 1
        if not config.skip_checkpoint and (local_step % (config.save_checkpoints_steps * args.gradient_accumulation_steps) == 0):
            saved_ckpt = True
            if is_main_process():
                save_path = manager.save(checkpoint_number=step)
                log(" ** Saved model checkpoint for step {}: {}".format(step, save_path))
            iter_save_path = iter_manager.save(checkpoint_number=step)
            log(" ** Saved iterator checkpoint for step {}: {}".format(step, iter_save_path), all_rank=True)

    step = (int(checkpoint.step) - 1)
    dllogger.flush()
    if not config.skip_checkpoint and not saved_ckpt:
        if is_main_process():
            save_path = manager.save(checkpoint_number=step)
            log(" ** Saved model checkpoint for step {}: {}".format(step, save_path))
        iter_save_path = iter_manager.save(checkpoint_number=step)
        log(" ** Saved iterator checkpoint for step {}: {}".format(step, iter_save_path), all_rank=True)

    return args


if __name__ == "__main__":
    start_time = time.time()
    args = main(start_time)
    log("Total Time:{:.4f}".format(time.time() - start_time))
    if is_main_process():
        postprocess_dllog(args)
