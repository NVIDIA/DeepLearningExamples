from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf
import horovod.tensorflow as hvd
import model
import data_utils
import lamb
import dllogger
from exp_utils import AverageMeter, setup_dllogger

import numpy as np

flags.DEFINE_integer("num_core_per_host", default=8,
      help="Number of cores per host")
flags.DEFINE_bool('horovod', True, 'Use Horovod ')
# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("raport_file", default="summary.json",
      help="Path to dlloger json")
flags.DEFINE_string("data_dir", default="",
      help="Path to tf-records directory.")
flags.DEFINE_string("record_info_dir", default="",
      help="Path to local directory containing filenames.txt.")
flags.DEFINE_string("corpus_info_path", default="",
      help="Path to corpus-info.json file.")
flags.DEFINE_string("model_dir", default="LM-TFM",
      help="Estimator model_dir.")
flags.DEFINE_bool("do_train", default=True,
      help="Whether to run training.")
flags.DEFINE_bool("do_eval", default=False,
      help="Whether to run eval on the dev set.")
flags.DEFINE_string("eval_ckpt_path", None,
      help="Checkpoint path for do_test evaluation."
           "If set, model_dir will be ignored."
           "If unset, will use the latest ckpt in model_dir.")
flags.DEFINE_bool("fp16", default=False,
      help="Whether to enable AMP ops.")
flags.DEFINE_bool("jit_optimizer", default=True,
      help="Whether to enable XLA on optimizer")

# Optimization config
flags.DEFINE_float("learning_rate", default=0.01,
      help="Maximum learning rate.")
flags.DEFINE_float("clip", default=0.25,
      help="Gradient clipping value.")
# for cosine decay
flags.DEFINE_float("min_lr_ratio", default=0.1,
      help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps", default=1000,
      help="Number of steps for linear lr warmup.")

# Training config
flags.DEFINE_integer("train_batch_size", default=256,
      help="Size of train batch.")
flags.DEFINE_integer("eval_batch_size", default=16,
      help="Size of valid batch.")
flags.DEFINE_integer("train_steps", default=40000,
      help="Total number of training steps.")
flags.DEFINE_integer("log_interval", default=100,
      help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=5000,
      help="number of steps for model checkpointing.")
flags.DEFINE_integer("batch_chunk", default=1,
      help="Number of accumulation steps.")

# Evaluation config
flags.DEFINE_integer("max_eval_batch", default=-1,
      help="Set -1 to turn off. Only used in test mode.")
flags.DEFINE_string("eval_split", "valid",
      help="Which data split to evaluate.")
flags.DEFINE_list("percentiles", default=['90', '95', '99'],
      help="percentiles for latency confidence intervals")

# Model config
flags.DEFINE_integer("tgt_len", default=192,
      help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=192,
      help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
      help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")

flags.DEFINE_integer("n_layer", default=16,
      help="Number of layers.")
flags.DEFINE_integer("d_model", default=512,
      help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=512,
      help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=8,
      help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=64,
      help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=2048,
      help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.1,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.0,
      help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
      help="untie r_w_bias and r_r_bias")

# Adaptive Softmax / Embedding
flags.DEFINE_bool("tie_weight", default=True,
      help="Tie embedding and softmax weight.")
flags.DEFINE_integer("div_val", default=1,
      help="Divide the embedding size by this val for each bin")
flags.DEFINE_bool("proj_share_all_but_first", default=False,
      help="True to share all but first projs, False not to share.")
flags.DEFINE_bool("proj_same_dim", default=True,
      help="Project the bin with the same dimension.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("proj_init_std", default=0.01,
      help="Initialization std for embedding projection.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")


FLAGS = flags.FLAGS

def get_model_fn(n_token, cutoffs):
  def model_fn(inp, tgt, mems, is_training):
    inp = tf.transpose(inp, [1, 0])
    tgt = tf.transpose(tgt, [1, 0])

    if FLAGS.init == "uniform":
      initializer = tf.initializers.random_uniform(
          minval=-FLAGS.init_range,
          maxval=FLAGS.init_range,
          seed=None)
    elif FLAGS.init == "normal":
      initializer = tf.initializers.random_normal(
          stddev=FLAGS.init_std,
          seed=None)
      proj_initializer = tf.initializers.random_normal(
          stddev=FLAGS.proj_init_std,
          seed=None)

    tie_projs = [False for _ in range(len(cutoffs) + 1)]
    if FLAGS.proj_share_all_but_first:
      for i in range(1, len(tie_projs)):
        tie_projs[i] = True

    loss, new_mems = model.transformer(
        dec_inp=inp,
        target=tgt,
        mems=mems,
        n_token=n_token,
        n_layer=FLAGS.n_layer,
        d_model=FLAGS.d_model,
        d_embed=FLAGS.d_embed,
        n_head=FLAGS.n_head,
        d_head=FLAGS.d_head,
        d_inner=FLAGS.d_inner,
        dropout=FLAGS.dropout,
        dropatt=FLAGS.dropatt,
        initializer=initializer,
        proj_initializer=proj_initializer,
        is_training=is_training,
        mem_len=FLAGS.mem_len,
        cutoffs=cutoffs,
        div_val=FLAGS.div_val,
        tie_projs=tie_projs,
        input_perms=None,
        target_perms=None,
        head_target=None,
        same_length=FLAGS.same_length,
        clamp_len=FLAGS.clamp_len,
        untie_r=FLAGS.untie_r,
        proj_same_dim=FLAGS.proj_same_dim)

    # number of parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('#params: {}'.format(num_params))

    if is_training:
      all_vars = tf.trainable_variables()

      return loss, new_mems, all_vars
    else:
      return loss, new_mems

  return model_fn


def single_core_graph(n_token, cutoffs, is_training, inp, tgt, mems):
  model_fn = get_model_fn(
      n_token=n_token,
      cutoffs=cutoffs)

  model_ret = model_fn(
      inp=inp,
      tgt=tgt,
      mems=mems,
      is_training=is_training)

  return model_ret


def train(n_token, cutoffs, rank, local_rank, size):

  meters = {}
  warmup = 2 + 12/size
  meters['train_throughput'] = AverageMeter(warmup=warmup)
  train_batch_size = FLAGS.train_batch_size // FLAGS.batch_chunk
  ##### Get input function and model function
  train_input_fn, train_record_info = data_utils.get_input_fn(
      record_info_dir=FLAGS.record_info_dir,
      split="train",
      per_host_bsz=train_batch_size,
      tgt_len=FLAGS.tgt_len,
      num_core_per_host=FLAGS.num_core_per_host,
      num_hosts=1)

  tf.logging.info("num of batches {}".format(train_record_info["num_batch"]))

  ##### Create computational graph
  train_set = train_input_fn({
      "batch_size": train_batch_size,
      "data_dir": FLAGS.data_dir})

  inputs, labels = train_set.make_one_shot_iterator().get_next()

  per_core_bsz = train_batch_size // FLAGS.num_core_per_host

  with tf.variable_scope(tf.get_variable_scope()):
    mems = [tf.Variable(tf.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], tf.float32), trainable=False)
              for _ in range(FLAGS.n_layer)]

    loss, new_mems, all_vars = single_core_graph(
        n_token=n_token,
        cutoffs=cutoffs,
        is_training=True,
        inp=inputs,
        tgt=labels,
        mems=mems)

    assign_mems = [mems[i].assign(new_mems[i]) for i in range(FLAGS.n_layer)]

  target_tokens = tf.size(labels)

  ## configure the optimizer
  global_step = tf.train.get_or_create_global_step()

  # warmup stage: increase the learning rate linearly
  if FLAGS.warmup_steps > 0:
    warmup_lr = tf.to_float(global_step) / tf.to_float(FLAGS.warmup_steps) \
                * FLAGS.learning_rate
  else:
    warmup_lr = 0.0

  # decay stage: decay the learning rate using the cosine schedule
  decay_lr = tf.train.cosine_decay(
      FLAGS.learning_rate,
      global_step=global_step-FLAGS.warmup_steps,
      decay_steps=FLAGS.train_steps-FLAGS.warmup_steps,
      alpha=FLAGS.min_lr_ratio)

  # choose warmup or decay
  learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                           warmup_lr, decay_lr)

  # get the train op
  optimizer = lamb.LAMBOptimizer(learning_rate=learning_rate)
  if FLAGS.horovod:
    optimizer = hvd.DistributedOptimizer(optimizer, sparse_as_dense=True)
  grads_and_vars = optimizer.compute_gradients(loss/FLAGS.batch_chunk, all_vars)
  grads, all_vars = zip(*grads_and_vars)

  accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in all_vars]
  in_progress = tf.get_variable(name="in_progress", shape=[], dtype=tf.bool, trainable=False,
                               initializer=tf.zeros_initializer)
  accum_ops = tf.cond(in_progress,
                      lambda: [accum_vars[i].assign_add(grad) for i, grad in enumerate(grads)],
                      lambda: [accum_vars[i].assign(grad) for i, grad in enumerate(grads)])
  with tf.control_dependencies(accum_ops + assign_mems):
    acc_op = in_progress.assign(tf.ones_like(in_progress))
  final_accum_vars = [accum_vars[i] + gv for i,gv in enumerate(grads)]
  acc_clipped, acc_gnorm = tf.clip_by_global_norm(final_accum_vars, FLAGS.clip)
  clipped, gnorm = tf.clip_by_global_norm(grads, FLAGS.clip)
  acc_train_op = optimizer.apply_gradients(list(zip(acc_clipped, all_vars)), global_step)
  grads_and_vars = list(zip(clipped, all_vars))
  if FLAGS.jit_optimizer:
    jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
    with jit_scope():
      train_op = optimizer.apply_gradients(grads_and_vars, global_step)
  else:
    train_op = optimizer.apply_gradients(grads_and_vars, global_step)
  final_op = tf.group(train_op, assign_mems)
  acc_final_op = tf.group(acc_train_op, assign_mems, in_progress.assign(tf.zeros_like(in_progress)))
  ##### Training loop
  saver = tf.train.Saver()

  gpu_options = tf.GPUOptions(allow_growth = True, visible_device_list = str(local_rank))
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options = gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    if FLAGS.horovod:
      sess.run(hvd.broadcast_global_variables(0))

    accum = [acc_op, target_tokens]
    fetches = [loss, global_step, target_tokens, learning_rate, final_op if FLAGS.batch_chunk == 1 else acc_final_op]
    total_loss, prev_step, target_tokens = 0., -1, 0
    start_time = time.time()
    while True:
      for i in range(FLAGS.batch_chunk-1):
        _,tt = sess.run(accum)
        target_tokens += tt
      fetched = sess.run(fetches)

      loss_np, curr_step, tt = fetched[:3]
      total_loss += loss_np
      target_tokens += tt

      if curr_step > 0 and curr_step % FLAGS.log_interval == 0:
        curr_loss = total_loss / (curr_step - prev_step)
        throughput = target_tokens * size / (time.time()-start_time)
        meters['train_throughput'].update(throughput)
        if rank == 0:
          tf.logging.info("step {} | lr {:8.9f} "
                        "| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}, tok/s {:>6.0f}".format(
                            curr_step, fetched[-2],
                            curr_loss, math.exp(curr_loss), curr_loss / math.log(2), throughput))
          dllogger_data = {
              'lr': fetched[-1],
              'train_loss': curr_loss,
              'train_perplexity': math.exp(curr_loss),
              'train_throughput': throughput,
          }
          dllogger.log(step=int(curr_step), data=dllogger_data)
        total_loss, prev_step, target_tokens = 0., curr_step, 0
        start_time = time.time()

      if curr_step > 0 and curr_step % FLAGS.save_steps == 0 and rank == 0:
        save_path = os.path.join(FLAGS.model_dir, "model.ckpt")
        saver.save(sess, save_path)
        tf.logging.info("Model saved in path: {}".format(save_path))

      if curr_step == FLAGS.train_steps:
        break
    if rank == 0:
      tf.logging.info("Training throughput: {:>6.0f} tok/s".format(meters['train_throughput'].avg))
      summary = {
          'train_throughput': meters['train_throughput'].avg,
      }
      dllogger.log(step=tuple(), data=summary)



def evaluate(n_token, cutoffs):
  ##### Get input function and model function
  eval_input_fn, eval_record_info = data_utils.get_input_fn(
      record_info_dir=FLAGS.record_info_dir,
      split=FLAGS.eval_split,
      per_host_bsz=FLAGS.eval_batch_size,
      tgt_len=FLAGS.tgt_len,
      num_core_per_host=FLAGS.num_core_per_host,
      num_hosts=1)

  meters = {}
  warmup = 2
  meters['eval_throughput'] = AverageMeter(warmup=warmup)
  meters['eval_latency'] = AverageMeter(warmup=warmup, keep=True)

  num_batch = eval_record_info["num_batch"]
  if FLAGS.max_eval_batch > 0:
      num_batch = FLAGS.max_eval_batch
  tf.logging.info("num of batches {}".format(num_batch))

  ##### Create computational graph
  eval_set = eval_input_fn({
      "batch_size": FLAGS.eval_batch_size,
      "data_dir": FLAGS.data_dir})

  inputs, labels = eval_set.make_one_shot_iterator().get_next()

  bsz = FLAGS.eval_batch_size

  with tf.variable_scope(tf.get_variable_scope()):
    mems = [tf.placeholder(tf.float32,
                             [FLAGS.mem_len, bsz, FLAGS.d_model])
              for _ in range(FLAGS.n_layer)]

    loss, new_mems = single_core_graph(
        n_token=n_token,
        cutoffs=cutoffs,
        is_training=False,
        inp=inputs,
        tgt=labels,
        mems=mems)

  target_tokens = tf.size(labels)
  ##### Evaluation loop
  mems_np = [np.zeros([FLAGS.mem_len, bsz, FLAGS.d_model], dtype=np.float32)
          for layer in range(FLAGS.n_layer)]

  saver = tf.train.Saver()

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    if FLAGS.eval_ckpt_path is None:
      eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    else:
      eval_ckpt_path = FLAGS.eval_ckpt_path
    tf.logging.info("Evaluate {}".format(eval_ckpt_path))
    saver.restore(sess, eval_ckpt_path)

    fetches = [loss, new_mems, target_tokens]

    format_str = "  >> processing batch {{:{0}d}}/{{:{0}d}}".format(
        len(str(num_batch)))

    total_loss, total_cnt, target_tokens = 0, 0, 0
    start_time = time.time()
    for step in range(num_batch):
      feed_dict = {}
      for m, m_np in zip(mems, mems_np):
        feed_dict[m] = m_np

      fetched = sess.run(fetches, feed_dict=feed_dict)

      loss_np, mems_np, tt = fetched
      target_tokens += tt
      cnt_np = 1
      total_loss += loss_np * cnt_np
      total_cnt += cnt_np

      elapsed = time.time()-start_time
      throughput = target_tokens / elapsed
      latency = elapsed*1000
      meters['eval_throughput'].update(throughput)
      meters['eval_latency'].update(latency)
      target_tokens = 0
      if (step+1) % (num_batch // 10) == 0:
        tf.logging.info(format_str.format(step+1, num_batch))
        dllogger_data = {
            'eval_latency': latency,
            'eval_throughput': throughput,
        }
        dllogger.log(step=step+1, data=dllogger_data)


      start_time = time.time()
    avg_loss = total_loss / total_cnt
    latency_data = np.array(meters['eval_latency'].vals)
    tf.logging.info("Evaluating with: bs {}, math {} ".format(FLAGS.eval_batch_size, "fp16" if FLAGS.fp16 else "fp32"))
    tf.logging.info("| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}, tok/s {:>6.1f}, ms/batch {:>4.2f}".format(
        avg_loss, math.exp(avg_loss), avg_loss / math.log(2), meters['eval_throughput'].avg, meters['eval_latency'].avg))
    summary = {
        'eval_loss': avg_loss,
        'eval_ppl': math.exp(avg_loss),
        'eval_avg_throughput': meters['eval_throughput'].avg,
        'eval_avg_latency': meters['eval_latency'].avg,
    }
    for p in FLAGS.percentiles:
      p = int(p)
      tf.logging.info("Latency {}%: {:>4.2f} ms".format(
        p, np.percentile(latency_data, p)))
      summary[f'eval_{p}%_latency'] = np.percentile(latency_data, p)
    dllogger.log(step=tuple(), data=summary)



def main(unused_argv):
  rank, local_rank, size = 0, 0, 1
  if FLAGS.horovod:
    hvd.init()
    rank = hvd.rank()
    local_rank = hvd.local_rank()
    size = hvd.size()
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.fp16:
      os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"
  else:
      os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "0"

  # Get corpus info
  corpus_info = data_utils.get_corpus_info(FLAGS.corpus_info_path)
  n_token = corpus_info["vocab_size"]
  cutoffs = corpus_info["cutoffs"][1:-1]
  tf.logging.info("n_token {}".format(n_token))

  setup_dllogger(enabled=True, filename=FLAGS.raport_file, rank=rank)

  if FLAGS.do_train:
    train(n_token, cutoffs, rank, local_rank, size)
  if FLAGS.do_eval:
    evaluate(n_token, cutoffs)



if __name__ == "__main__":
  tf.app.run()
