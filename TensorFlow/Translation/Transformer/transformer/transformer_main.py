# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
# ==============================================================================
"""Creates an estimator to train the Transformer model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import sys
import tempfile
import random
import numpy.random

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import horovod.tensorflow as hvd
from mpi4py import MPI
import sacrebleu
import numpy as np
import sentencepiece as spm

from model import transformer
from model import model_params, model_utils
import translate
from utils import dataset
from utils import metrics
from utils import tokenizer
from utils import distributed_utils
from utils import compute_bleu
from options import get_parser

import model.mixed_precision_optimizer as mpo
from pynvml import *
from numa import *

DEFAULT_TRAIN_EPOCHS = 10
BLEU_DIR = "bleu"
INF = 10000

# report samples/sec, total loss and learning rate during training
class _LogSessionRunHook(tf.train.SessionRunHook):
  def __init__(self, global_batch_size, display_every=10):
    self.global_batch_size = global_batch_size
    self.display_every = display_every

  def after_create_session(self, session, coord):
    print('|  Step words/sec   Loss  Learning-rate')
    self.elapsed_secs = 0.
    self.count = 0

  def before_run(self, run_context):
    self.t0 = time.time()
    return tf.train.SessionRunArgs(
        fetches=['global_step:0', 'model/masked_loss:0',
                 'model/get_train_op/learning_rate/learning_rate:0'])

  def after_run(self, run_context, run_values):
    self.elapsed_secs += time.time() - self.t0
    self.count += 1
    global_step, loss, lr = run_values.results
    total_loss = 0.0
    if FLAGS.enable_horovod:
        comm = MPI.COMM_WORLD
        losses = comm.allgather(loss)
        total_loss = np.mean(losses)
    else:
        total_loss = loss
    print_step = global_step + 1 # One-based index for printing.
    if not FLAGS.enable_horovod or hvd.rank() == 0:
        if print_step == 1 or print_step % self.display_every == 0:
            dt = self.elapsed_secs / self.count
            img_per_sec = self.global_batch_size / dt
            print('|%6i %9.1f %6.3f     %6.4e' %
                  (print_step, img_per_sec, total_loss, lr))
            self.elapsed_secs = 0.
            self.count = 0


def model_fn(features, labels, mode, params):
    """Defines how to train, evaluate and predict from the transformer model."""
    with tf.variable_scope("model"):
        inputs, targets = features, labels

        # Create model and get output logits.
        model = transformer.Transformer(params, mode == tf.estimator.ModeKeys.TRAIN)

        output = model(inputs, targets)

        # When in prediction mode, the labels/targets is None. The model output
        # is the prediction
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                tf.estimator.ModeKeys.PREDICT,
                predictions=output)

        logits = output

        # Calculate model loss.
        xentropy, weights = metrics.padded_cross_entropy_loss(
            logits, targets, params.label_smoothing, params.vocab_size)

        loss = tf.divide(tf.reduce_sum(xentropy * weights), tf.reduce_sum(weights), name='masked_loss')

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, predictions={"predictions": logits},
                eval_metric_ops=metrics.get_eval_metrics(logits, labels, params))
        else:
            train_op = get_train_op(loss, params)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.cast(learning_rate_warmup_steps, tf.float32)
        step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)

        learning_rate *= (hidden_size ** -0.5)
        # Apply linear warmup
        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        # Apply rsqrt decay
        learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))
        if FLAGS.enable_horovod:
            learning_rate *= hvd.size()
        learning_rate = tf.identity(learning_rate, name='learning_rate') #for logging purposes

        # Save learning rate value to TensorBoard summary.
        tf.summary.scalar("learning_rate", learning_rate)

        return learning_rate


def get_train_op(loss, params):
    """Generate training operation that updates variables based on loss."""
    with tf.variable_scope("get_train_op"):
        learning_rate = get_learning_rate(
            params.learning_rate, params.hidden_size,
            params.learning_rate_warmup_steps)

        # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster
        # than the TF core Adam optimizer.
        optimizer = tf.contrib.opt.LazyAdamOptimizer(
            learning_rate,
            beta1=params.optimizer_adam_beta1,
            beta2=params.optimizer_adam_beta2,
            epsilon=params.optimizer_adam_epsilon)
  
        if FLAGS.enable_horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, sparse_as_dense=True)
        if FLAGS.enable_amp:
            loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(init_loss_scale=2**32,
                                                                                              incr_every_n_steps=1000,
                                                                                              decr_every_n_nan_or_inf=2,
                                                                                              decr_ratio=0.5)
            optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(optimizer, loss_scale_manager)

        # Calculate and apply gradients using LazyAdamOptimizer.
        global_step = tf.train.get_global_step()
        tvars = tf.trainable_variables()
        gradients = optimizer.compute_gradients(
            loss, tvars, colocate_gradients_with_ops=True)
        train_op = optimizer.apply_gradients(
            gradients, global_step=global_step, name="train")

        # Save gradient norm to Tensorboard
        tf.summary.scalar("global_norm/gradient_norm",
                          tf.global_norm(list(zip(*gradients))[0]))

        return train_op

def translate_and_compute_bleu(params, estimator, subtokenizer, bleu_source, bleu_ref):
    """Translate file and report the cased and uncased bleu scores."""
    # Create temporary file on worker 0 to store translation.
    if not params.enable_horovod or hvd.rank() == 0:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp_filename = tmp.name
    else:
        tmp, tmp_filename = None, None

    translate.translate_file(
            params,
            estimator,
            subtokenizer,
            bleu_source,
            output_file=tmp_filename,
            print_all_translations=False
            )

    if not params.enable_horovod or hvd.rank() == 0:
        # Compute uncased and cased bleu scores.
        uncased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, False)
        cased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, True)

        #SacreBleu evaluation
        ref_lines = [tf.gfile.Open(tmp_filename).read().strip().splitlines()]
        hyp_lines = tf.gfile.Open(tmp_filename).read().strip().splitlines()

        sb_uncased_score = sacrebleu.corpus_bleu(hyp_lines, ref_lines, lowercase=True).score
        sb_cased_score = sacrebleu.corpus_bleu(hyp_lines, ref_lines, lowercase=False).score

        os.remove(tmp_filename)
    else:
        uncased_score, cased_score, sb_uncased_score, sb_cased_score = 0.0, 0.0, 0.0, 0.0

    if params.enable_horovod:
        comm = MPI.COMM_WORLD
        uncased_score    = comm.bcast(uncased_score, root=0)
        cased_score      = comm.bcast(cased_score, root=0)
        sb_uncased_score = comm.bcast(sb_uncased_score, root=0)
        sb_cased_score   = comm.bcast(sb_uncased_score, root=0)
    return uncased_score, cased_score, sb_uncased_score, sb_cased_score

def get_global_step(estimator):
    """Return estimator's last checkpoint."""
    return int(estimator.latest_checkpoint().split("-")[-1])


def evaluate_and_log_bleu(params, estimator, bleu_writer, bleu_source, bleu_ref):
    """Calculate and record the BLEU score."""
    if params.sentencepiece:
        subtokenizer = spm.SentencePieceProcessor()
        subtokenizer.load('{}.model'.format(os.path.join(FLAGS.data_dir, FLAGS.vocab_file)))
    else:
        subtokenizer = tokenizer.Subtokenizer(
            os.path.join(FLAGS.data_dir, FLAGS.vocab_file), reserved_tokens='assumed_in_file')
  
    uncased_score, cased_score, sb_uncased_score, sb_cased_score = translate_and_compute_bleu(params, estimator, subtokenizer, bleu_source, bleu_ref)

    print("Bleu score (uncased):", uncased_score)
    print("Bleu score (cased):", cased_score)
    print("SacreBleu score (uncased):", sb_uncased_score)
    print("SacreBleu score (cased):", sb_cased_score)

    summary = tf.Summary(value=[
        tf.Summary.Value(tag="bleu/uncased", simple_value=uncased_score),
        tf.Summary.Value(tag="bleu/cased", simple_value=cased_score),
    ])

    bleu_writer.add_summary(summary, get_global_step(estimator))
    bleu_writer.flush()
    return uncased_score, cased_score


def train_schedule(
        estimator, params, train_eval_iterations, single_iteration_train_steps=None,
        bleu_source=None, bleu_ref=None, bleu_threshold=None):
    """Train and evaluate model, and optionally compute model's BLEU score.

    **Step vs. Epoch vs. Iteration**

    Steps and epochs are canonical terms used in TensorFlow and general machine
    learning. They are used to describe running a single process (train/eval):
      - Step refers to running the process through a single or batch of examples.
      - Epoch refers to running the process through an entire dataset.

    E.g. training a dataset with 100 examples. The dataset is
    divided into 20 batches with 5 examples per batch. A single training step
    trains the model on one batch. After 20 training steps, the model will have
    trained on every batch in the dataset, or, in other words, one epoch.

    Meanwhile, iteration is used in this implementation to describe running
    multiple processes (training and eval).
      - A single iteration:
        1. trains the model for a specific number of steps or epochs.
        2. evaluates the model.
        3. (if source and ref files are provided) compute BLEU score.

    This function runs through multiple train+eval+bleu iterations.

    Args:
      estimator: tf.Estimator containing model to train.
      train_eval_iterations: Number of times to repeat the train+eval iteration.
      single_iteration_train_steps: Number of steps to train in one iteration.
      bleu_source: File containing text to be translated for BLEU calculation.
      bleu_ref: File containing reference translations for BLEU calculation.
      bleu_threshold: minimum BLEU score before training is stopped.
    """
    evaluate_bleu = bleu_source is not None and bleu_ref is not None

    # Print out training schedule
    print("Training schedule:")
    print("\t1. Train for %d steps." % single_iteration_train_steps)
    print("\t2. Evaluate model.")
    if evaluate_bleu:
        print("\t3. Compute BLEU score.")
        if bleu_threshold is not None:
            print("Repeat above steps until the BLEU score reaches", bleu_threshold)
            # Change loop stopping condition if bleu_threshold is defined.
            train_eval_iterations = INF
    if not evaluate_bleu or bleu_threshold is None:
        print("Repeat above steps %d times." % train_eval_iterations)

    if evaluate_bleu:
        # Set summary writer to log bleu score.
        bleu_writer = tf.summary.FileWriter(
            os.path.join(estimator.model_dir, BLEU_DIR))

    training_hooks = []
    if FLAGS.enable_horovod:
        training_hooks.append(hvd.BroadcastGlobalVariablesHook(0))
    if FLAGS.report_loss:
        global_batch_size = estimator.params.batch_size if not FLAGS.enable_horovod else estimator.params.batch_size*hvd.size()
        training_hooks.append(_LogSessionRunHook(global_batch_size, params.display_interval))
    # training_hooks.append(LogTrainRunHook(-1 if not FLAGS.enable_horovod else hvd.rank()))
    # Loop training/evaluation/bleu cycles
    for i in xrange(train_eval_iterations):
        print("Starting iteration", i + 1)

        # Train the model for single_iteration_train_steps or until the input fn
        # runs out of examples (if single_iteration_train_steps is None).
        estimator.train(dataset.train_input_fn, hooks=training_hooks, steps=single_iteration_train_steps)

        if not FLAGS.enable_horovod or hvd.rank() == 0:
            eval_results = estimator.evaluate(dataset.eval_input_fn)
            print("Evaluation results (iter %d/%d):" % (i + 1, train_eval_iterations),
                  eval_results)

        if evaluate_bleu:
            uncased_score, _ = evaluate_and_log_bleu(
                params, estimator, bleu_writer, bleu_source, bleu_ref)
            if bleu_threshold is not None and uncased_score > bleu_threshold:
                bleu_writer.close()
                break

    print("Training finished!")

def main(_):
    # Set logging level to INFO to display training progress (logged by the
    # estimator)
    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.enable_horovod:
        hvd.init()
        distributed_utils.suppress_output()

    # Set random seed.
    if FLAGS.random_seed is None:
        raise Exception('No Random seed given')
    print('Setting random seed = ', FLAGS.random_seed)
    seed = FLAGS.random_seed
    random.seed(seed)
    tf.set_random_seed(seed)
    numpy.random.seed(seed)

    if FLAGS.params == "base":
        params = model_params.TransformerBaseParams
    elif FLAGS.params == "big":
        params = model_params.TransformerBigParams
    else:
        raise ValueError("Invalid parameter set defined: %s."
                         "Expected 'base' or 'big.'" % FLAGS.params)

    # Determine training schedule based on flags.
    train_eval_iterations = FLAGS.train_steps // FLAGS.steps_between_eval
    single_iteration_train_steps = FLAGS.steps_between_eval
    # Make sure that the BLEU source and ref files if set
    if FLAGS.bleu_source is not None and FLAGS.bleu_ref is not None:
        if not tf.gfile.Exists(FLAGS.bleu_source):
            raise ValueError("BLEU source file %s does not exist" % FLAGS.bleu_source)
        if not tf.gfile.Exists(FLAGS.bleu_ref):
            raise ValueError("BLEU source file %s does not exist" % FLAGS.bleu_ref)
    # Make sure that vocab file has the same number of tokens as model output
    vocab_file = os.path.join(FLAGS.data_dir, FLAGS.vocab_file)
    vocab_file += '.vocab' if FLAGS.sentencepiece else ''
    vocab_file_size = sum(1 for line in open(vocab_file, 'r', newline='\n'))
    if  vocab_file_size != params.vocab_size:
        raise ValueError("Vocab file has different lenght than declared vocab size. Vocab file size: {}".format(vocab_file_size))

    # Add flag-defined parameters to params object
    params.data_dir = FLAGS.data_dir
    params.enable_horovod = FLAGS.enable_horovod
    params.num_cpu_cores = FLAGS.num_cpu_cores
    if FLAGS.batch_size > 0:
        params.batch_size = FLAGS.batch_size
    if float(FLAGS.learning_rate) > 0:
        params.learning_rate = float(FLAGS.learning_rate)
    params.learning_rate_warmup_steps = FLAGS.warmup_steps
    params.sentencepiece = FLAGS.sentencepiece

    if params.sentencepiece:
        subtokenizer = spm.SentencePieceProcessor()
        subtokenizer.load('{}.model'.format(os.path.join(FLAGS.data_dir, FLAGS.vocab_file)))
    else:
        subtokenizer = tokenizer.Subtokenizer(
            os.path.join(FLAGS.data_dir, FLAGS.vocab_file), reserved_tokens='assumed_in_file')
    params.eos_id = subtokenizer.eos_id()

    config = tf.ConfigProto()
    if FLAGS.enable_horovod: 
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        # set CPU affinity mask to what NVML is recommending for optimal performance
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(hvd.local_rank())
        cpuSet = nvmlDeviceGetCpuAffinity(handle, 4)
        cpuset = []
    
        for i in range(cpuSet._length_ * 64):
            word = i // 64
            bit = i % 64
      
            if cpuSet[word] & (1 << bit) != 0:
                cpuset.append(i)
    
        set_affinity(os.getpid(), cpuset)
        nvmlShutdown()

    if FLAGS.enable_xla:
        config.gpu_options.allow_growth = True 
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    if FLAGS.enable_amp:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE'] = '1'

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        session_config=config,
        keep_checkpoint_max=3,
        save_checkpoints_steps=2000 if not FLAGS.enable_horovod or hvd.rank() == 0 else None,
        log_step_count_steps=100,
        tf_random_seed=FLAGS.random_seed,
        )


    print("\n******Using Configuration********")
    print(vars(params))
    print(FLAGS)
    print("*********************************\n")
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, 
        model_dir=FLAGS.model_dir, 
        config=run_config,
        params=params)

    train_schedule(
        estimator, params, train_eval_iterations, single_iteration_train_steps,
        FLAGS.bleu_source, FLAGS.bleu_ref, FLAGS.bleu_threshold)

if __name__ == "__main__":
    parser = get_parser()
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
