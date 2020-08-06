# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
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

import os
import sys
import subprocess
import time
import argparse
import json
import logging

import tensorflow as tf

import horovod.tensorflow as hvd
from horovod.tensorflow.compression import Compression

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from tqdm import tqdm
import dllogger
from utils import is_main_process, format_step, get_rank, get_world_size
from configuration import ElectraConfig
from modeling import TFElectraForQuestionAnswering
from tokenization import ElectraTokenizer
from optimization import create_optimizer
from squad_utils import SquadV1Processor, SquadV2Processor, squad_convert_examples_to_features, \
    SquadResult, RawResult, get_answers

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/electra-small-generator",
    "google/electra-base-generator",
    "google/electra-large-generator",
    "google/electra-small-discriminator",
    "google/electra-base-discriminator",
    "google/electra-large-discriminator",
    # See all ELECTRA models at https://huggingface.co/models?filter=electra
]

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--electra_model", default=None, type=str, required=True,
                        help="Model selected in the list: " + ", ".join(TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST))
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="Path to dataset.")
    parser.add_argument("--output_dir", default=".", type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        #                    required=True,
                        help="The checkpoint file from pretraining")

    # Other parameters
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to use evaluate accuracy of predictions")
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--layerwise_lr_decay", default=0.8, type=float,
                        help="The layerwise learning rate decay. Shallower layers have lower learning rates.")

    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1.0, type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")

    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument(
        "--joint_head",
        default=True,
        type=bool,
        help="Jointly predict the start and end positions",
    )
    parser.add_argument(
        "--beam_size",
        default=4,
        type=int,
        help="Beam size when doing joint predictions",
    )
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")

    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--amp',
                        action='store_true',
                        help="Automatic mixed precision training")
    parser.add_argument('--fp16_all_reduce',
                        action='store_true',
                        help="Whether to use 16-bit all reduce")
    parser.add_argument('--xla',
                        action='store_true',
                        help="Whether to use XLA")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--log_freq',
                        type=int, default=50,
                        help='frequency of logging loss.')
    parser.add_argument('--json-summary', type=str, default="results/dllogger.json",
                        help='If provided, the json summary will be written to the specified file.')
    parser.add_argument("--eval_script",
                        help="Script to evaluate squad predictions",
                        default="evaluate.py",
                        type=str)
    parser.add_argument("--use_env",
                        action='store_true',
                        help="Whether to read local rank from ENVVAR")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--disable-progress-bar',
                        default=False,
                        action='store_true',
                        help='Disable tqdm progress bar')
    parser.add_argument("--skip_cache",
                        default=False,
                        action='store_true',
                        help="Whether to cache train features")
    parser.add_argument("--cache_dir",
                        default=None,
                        type=str,
                        help="Location to cache train feaures. Will default to the dataset direct")
    args = parser.parse_args()

    if not args.do_train and (not args.init_checkpoint or args.init_checkpoint == 'None'):
        raise ValueError("Checkpoint is required if do_train is not set")

    return args


def get_dataset_from_features(features, batch_size, drop_remainder=True, ngpu=8, mode="train", v2=False):
    """Input function for training"""

    all_input_ids = tf.convert_to_tensor([f.input_ids for f in features], dtype=tf.int64)
    all_input_mask = tf.convert_to_tensor([f.attention_mask for f in features], dtype=tf.int64)
    all_segment_ids = tf.convert_to_tensor([f.token_type_ids for f in features], dtype=tf.int64)
    all_start_pos = tf.convert_to_tensor([f.start_position for f in features], dtype=tf.int64)
    all_end_pos = tf.convert_to_tensor([f.end_position for f in features], dtype=tf.int64)

    # if v2 else None:
    all_cls_index = tf.convert_to_tensor([f.cls_index for f in features], dtype=tf.int64)
    all_p_mask = tf.convert_to_tensor([f.p_mask for f in features], dtype=tf.float32)
    all_is_impossible = tf.convert_to_tensor([f.is_impossible for f in features], dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (all_input_ids, all_input_mask, all_segment_ids, all_start_pos, all_end_pos)
        + (all_cls_index, all_p_mask, all_is_impossible))
    if ngpu > 1:
        dataset = dataset.shard(get_world_size(), get_rank())

    if mode == "train":
        dataset = dataset.shuffle(batch_size * 3)
    # dataset = dataset.map(self._preproc_samples,
    #                      num_parallel_calls=multiprocessing.cpu_count()//self._num_gpus)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(batch_size)

    return dataset


@tf.function
def train_step(model, inputs, loss, amp, opt, init, v2=False, loss_class=None, fp16=False):
    with tf.GradientTape() as tape:
        [input_ids, input_mask, segment_ids, start_positions, end_positions, cls_index, p_mask, is_impossible] = inputs

        if not v2:
            is_impossible = None

        start_logits, end_logits, cls_logits = model(input_ids,
                                                     # input_ids=input_ids,
                                                     attention_mask=input_mask,
                                                     token_type_ids=segment_ids,
                                                     start_positions=start_positions,
                                                     end_positions=end_positions,
                                                     cls_index=cls_index,
                                                     p_mask=p_mask,
                                                     is_impossible=is_impossible,
                                                     position_ids=None,
                                                     head_mask=None,
                                                     inputs_embeds=None,
                                                     training=True,
                                                     )[0:3]

        # If we are on multi-GPU, split add a dimension
        if len(start_positions.shape) > 1:
            start_positions = tf.squeeze(start_positions, axis=-1, name="squeeze_start_positions")
        if len(end_positions.shape) > 1:
            end_positions = tf.squeeze(end_positions, axis=-1, name="squeeze_end_positions")
        if is_impossible is not None and len(is_impossible.shape) > 1 and v2 and cls_logits is not None:
            is_impossible = tf.squeeze(is_impossible, axis=-1, name="squeeze_is_impossible")

        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.shape[1]
        start_positions = tf.clip_by_value(start_positions, 0, ignored_index, name="clip_start_positions")
        end_positions = tf.clip_by_value(end_positions, 0, ignored_index, name="clip_end_positions")

        start_loss = loss(y_true=start_positions, y_pred=start_logits)
        end_loss = loss(y_true=end_positions, y_pred=end_logits)
        loss_value = (start_loss + end_loss) / 2

        if v2:
            cls_loss_value = loss_class(y_true=is_impossible, y_pred=cls_logits)
            loss_value += cls_loss_value * 0.5

        unscaled_loss = tf.stop_gradient(loss_value)
        if amp:
            loss_value = opt.get_scaled_loss(loss_value)

    tape = hvd.DistributedGradientTape(tape, sparse_as_dense=True,
                                       compression=Compression.fp16 if fp16 else Compression.none)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    if amp:
        gradients = opt.get_unscaled_gradients(gradients)
    opt.apply_gradients(zip(gradients, model.trainable_variables))  # , clip_norm=1.0)

    if init:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)

    return unscaled_loss  # , outputs#, tape.gradient(loss_value, model.trainable_variables)


@tf.function
def infer_step(model, input_ids,
               attention_mask=None,
               token_type_ids=None,
               cls_index=None,
               p_mask=None,
               position_ids=None,
               head_mask=None,
               inputs_embeds=None,
               training=False,
               ):
    return model(input_ids,
                 attention_mask=attention_mask,
                 token_type_ids=token_type_ids,
                 cls_index=cls_index,
                 p_mask=p_mask,
                 position_ids=position_ids,
                 head_mask=head_mask,
                 inputs_embeds=inputs_embeds,
                 training=training,
                 )


def main():
    args = parse_args()

    hvd.init()
    if is_main_process():
        print("Running total processes: {}".format(get_world_size()))
    print("Starting process: {}".format(get_rank()))

    if is_main_process():
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.json_summary),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)])
    else:
        dllogger.init(backends=[])

    tf.random.set_seed(args.seed)
    dllogger.log(step="PARAMETER", data={"SEED": args.seed})
    # script parameters
    BATCH_SIZE = args.train_batch_size
    EVAL_BATCH_SIZE = args.predict_batch_size
    USE_XLA = args.xla
    USE_AMP = args.amp
    EPOCHS = args.num_train_epochs

    if not args.do_train:
        EPOCHS = args.num_train_epochs = 1
        print("Since running inference only, setting args.num_train_epochs to 1")

    if not os.path.exists(args.output_dir) and is_main_process():
        os.makedirs(args.output_dir)

    # TensorFlow configuration
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tf.config.optimizer.set_jit(USE_XLA)
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

    if is_main_process():
        logger.info("***** Loading tokenizer and model *****")
    # Load tokenizer and model from pretrained model/vocabulary. Specify the number of labels to classify (2+: classification, 1: regression)
    electra_model = args.electra_model
    config = ElectraConfig.from_pretrained(electra_model, cache_dir=args.cache_dir)
    config.update({"amp": args.amp})
    tokenizer = ElectraTokenizer.from_pretrained(electra_model, cache_dir=args.cache_dir)
    model = TFElectraForQuestionAnswering.from_pretrained(electra_model, config=config, cache_dir=args.cache_dir, args=args)

    if is_main_process():
        logger.info("***** Loading dataset *****")
    # Load data
    processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
    train_examples = processor.get_train_examples(args.data_dir) if args.do_train else None
    dev_examples = processor.get_dev_examples(args.data_dir) if args.do_predict else None

    if is_main_process():
        logger.info("***** Loading features *****")
    # Load cached features
    squad_version = '2.0' if args.version_2_with_negative else '1.1'
    if args.cache_dir is None:
        args.cache_dir = args.data_dir
    cached_train_features_file = args.cache_dir.rstrip('/') + '/' + 'TF2_train-v{4}.json_{0}_{1}_{2}_{3}'.format(
        electra_model.split("/")[1], str(args.max_seq_length), str(args.doc_stride),
        str(args.max_query_length), squad_version)
    cached_dev_features_file = args.cache_dir.rstrip('/') + '/' + 'TF2_dev-v{4}.json_{0}_{1}_{2}_{3}'.format(
        electra_model.split("/")[1], str(args.max_seq_length), str(args.doc_stride),
        str(args.max_query_length), squad_version)

    try:
        with open(cached_train_features_file, "rb") as reader:
            train_features = pickle.load(reader) if args.do_train else []
        with open(cached_dev_features_file, "rb") as reader:
            dev_features = pickle.load(reader) if args.do_predict else []
    except:
        train_features = (  # TODO: (yy) do on rank 0?
            squad_convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True,
                return_dataset="",
            )
            if args.do_train
            else []
        )
        dev_features = (
            squad_convert_examples_to_features(
                examples=dev_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False,
                return_dataset="",
            )
            if args.do_predict
            else []
        )
        # Dump Cached features
        if not args.skip_cache and is_main_process():
            if args.do_train:
                print("***** Building Cache Files: {} *****".format(cached_train_features_file))
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)
            if args.do_predict:
                print("***** Building Cache Files: {} *****".format(cached_dev_features_file))
                with open(cached_dev_features_file, "wb") as writer:
                    pickle.dump(dev_features, writer)

    len_train_features = len(train_features)
    total_train_steps = int((len_train_features * EPOCHS / BATCH_SIZE) / get_world_size()) + 1
    train_steps_per_epoch = int((len_train_features / BATCH_SIZE) / get_world_size()) + 1
    len_dev_features = len(dev_features)
    total_dev_steps = int((len_dev_features / EVAL_BATCH_SIZE)) + 1

    train_dataset = get_dataset_from_features(train_features, BATCH_SIZE,
                                              v2=args.version_2_with_negative) if args.do_train else []
    dev_dataset = get_dataset_from_features(dev_features, EVAL_BATCH_SIZE, drop_remainder=False, ngpu=1, mode="dev",
                                            v2=args.version_2_with_negative) if args.do_predict else []

    opt = create_optimizer(init_lr=args.learning_rate, num_train_steps=total_train_steps,
                           num_warmup_steps=int(args.warmup_proportion * total_train_steps),
                           weight_decay_rate=args.weight_decay_rate,
                           layerwise_lr_decay=args.layerwise_lr_decay,
                           n_transformer_layers=model.num_hidden_layers)
    if USE_AMP:
        # loss scaling is currently required when using mixed precision
        opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")

    # Define loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_class = tf.keras.losses.BinaryCrossentropy(
        from_logits=True,
        name='binary_crossentropy'
    )
    metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
    model.compile(optimizer=opt, loss=loss, metrics=[metric])
    train_loss_results = []

    if args.do_train and is_main_process():
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len_train_features)
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * get_world_size(),
        )
        logger.info("  Total optimization steps = %d", total_train_steps)

    total_train_time = 0
    latency = []
    for epoch in range(EPOCHS):
        if args.do_train:
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_perf_avg = tf.keras.metrics.Mean()
            epoch_start = time.time()

            epoch_iterator = tqdm(train_dataset, total=train_steps_per_epoch, desc="Iteration", mininterval=5,
                                  disable=not is_main_process())
            for iter, inputs in enumerate(epoch_iterator):
                # breaking criterion if max_steps if > 1
                if args.max_steps > 0 and (epoch * train_steps_per_epoch + iter) > args.max_steps:
                    break
                iter_start = time.time()
                # Optimize the model
                loss_value = train_step(model, inputs, loss, USE_AMP, opt, (iter == 0 and epoch == 0),
                                        v2=args.version_2_with_negative, loss_class=loss_class, fp16=USE_AMP)
                epoch_perf_avg.update_state(1. * BATCH_SIZE / (time.time() - iter_start))
                if iter % 100 == 0:
                    if is_main_process():
                        print("Epoch: {:03d}, Step:{:6d}, Loss:{:12.8f}, Perf:{:5.0f}".format(epoch, iter, loss_value,
                                                                                              epoch_perf_avg.result() * get_world_size()))
                    dllogger.log(step=(epoch, iter,), data={"step_loss": float(loss_value.numpy()),
                                                            "train_perf": float( epoch_perf_avg.result().numpy() * get_world_size())})

                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss

            # End epoch
            train_loss_results.append(epoch_loss_avg.result())
            total_train_time += float(time.time() - epoch_start)
            # Summarize and save checkpoint at the end of each epoch
            if is_main_process():
                # print(
                #    "**TRAIN SUMMARY** - Epoch {:03d}, Train_Loss: {:12.8f}, Train_Perf: {:5.0f} seq/s, Train_Time: {:5.0f} s"
                #    .format(epoch, epoch_loss_avg.result(), epoch_perf_avg.result() * get_world_size(), total_train_time))

                dllogger.log(step=tuple(), data={"e2e_train_time": total_train_time,
                                                 "training_sequences_per_second": float(
                                                     epoch_perf_avg.result().numpy() * get_world_size()),
                                                 "final_loss": float(epoch_loss_avg.result().numpy())})

            if not args.skip_checkpoint:
                # checkpoint_name = "/workspace/electra/checkpoints/electra_base_qa_v2_{}_joint_head_{}_seed_{}_lr_{}_ckpt_{}".format(
                #     args.version_2_with_negative, args.joint_head, args.seed, args.learning_rate, epoch + 1)
                checkpoint_name = "checkpoints/electra_base_qa_v2_{}_epoch_{}_ckpt".format(args.version_2_with_negative, epoch + 1)
                if is_main_process():
                    model.save_weights(checkpoint_name)


        if args.do_predict and (args.evaluate_during_training or epoch == args.num_train_epochs - 1):
            if not args.do_train:
                logger.info("***** Loading checkpoint: {} *****".format(args.init_checkpoint))
                model.load_weights(args.init_checkpoint).expect_partial()

            current_feature_id = 0
            all_results = []
            if is_main_process():
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", total_dev_steps)
                logger.info("  Batch size = %d", args.predict_batch_size)

            raw_infer_start = time.time()
            if is_main_process():
                infer_perf_avg = tf.keras.metrics.Mean()
                dev_iterator = tqdm(dev_dataset, total=total_dev_steps, desc="Iteration", mininterval=5,
                                    disable=not is_main_process())
                for input_ids, input_mask, segment_ids, start_positions, end_positions, cls_index, p_mask, is_impossible in dev_iterator:
                    # training=False is needed only if there are layers with different
                    # behavior during training versus inference (e.g. Dropout).

                    iter_start = time.time()

                    if not args.joint_head:
                        batch_start_logits, batch_end_logits = infer_step(model, input_ids,
                                                                          attention_mask=input_mask,
                                                                          token_type_ids=segment_ids,
                                                                          )[:2]
                    else:
                        outputs = infer_step(model, input_ids,
                                             attention_mask=input_mask,
                                             token_type_ids=segment_ids,
                                             cls_index=cls_index,
                                             p_mask=p_mask,
                                             )

                    infer_time = (time.time() - iter_start)
                    infer_perf_avg.update_state(1. * EVAL_BATCH_SIZE / infer_time)
                    latency.append(1. * infer_time / EVAL_BATCH_SIZE)

                    for iter_ in range(input_ids.shape[0]):

                        if not args.joint_head:
                            start_logits = batch_start_logits[iter_].numpy().tolist()
                            end_logits = batch_end_logits[iter_].numpy().tolist()
                            dev_feature = dev_features[current_feature_id]
                            current_feature_id += 1
                            unique_id = int(dev_feature.unique_id)
                            all_results.append(RawResult(unique_id=unique_id,
                                                         start_logits=start_logits,
                                                         end_logits=end_logits))
                        else:
                            dev_feature = dev_features[current_feature_id]
                            current_feature_id += 1
                            unique_id = int(dev_feature.unique_id)
                            output = [output[iter_].numpy().tolist() for output in outputs]

                            start_logits = output[0]
                            start_top_index = output[1]
                            end_logits = output[2]
                            end_top_index = output[3]
                            cls_logits = output[4]
                            result = SquadResult(
                                unique_id,
                                start_logits,
                                end_logits,
                                start_top_index=start_top_index,
                                end_top_index=end_top_index,
                                cls_logits=cls_logits,
                            )

                            all_results.append(result)

                # Compute and save predictions
                answers, nbest_answers = get_answers(dev_examples, dev_features, all_results, args)

                output_prediction_file = os.path.join(args.output_dir, "predictions.json")
                output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
                e2e_infer_time = time.time() - raw_infer_start
                # if args.version_2_with_negative:
                #     output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
                # else:
                #     output_null_log_odds_file = None
                with open(output_prediction_file, "w") as f:
                    f.write(json.dumps(answers, indent=4) + "\n")
                with open(output_nbest_file, "w") as f:
                    f.write(json.dumps(nbest_answers, indent=4) + "\n")

                if args.do_eval:
                    if args.version_2_with_negative:
                        dev_file = "dev-v2.0.json"
                    else:
                        dev_file = "dev-v1.1.json"

                    eval_out = subprocess.check_output([sys.executable, args.eval_script,
                                                        args.data_dir + "/" + dev_file, output_prediction_file])
                    print(eval_out.decode('UTF-8'))
                    scores = str(eval_out).strip()
                    exact_match = float(scores.split(":")[1].split(",")[0])
                    if args.version_2_with_negative:
                        f1 = float(scores.split(":")[2].split(",")[0])
                    else:
                        f1 = float(scores.split(":")[2].split("}")[0])

                    logger.info("Epoch: {:03d} Results: {}".format(epoch, eval_out.decode('UTF-8')))
                    print("**EVAL SUMMARY** - Epoch: {:03d},  EM: {:6.3f}, F1: {:6.3f}, Infer_Perf: {:4.0f} seq/s"
                          .format(epoch, exact_match, f1, infer_perf_avg.result()))

                latency_all = sorted(latency)[:-2]
                print(
                    "**LATENCY SUMMARY** - Epoch: {:03d},  Ave: {:6.3f} ms, 90%: {:6.3f} ms, 95%: {:6.3f} ms, 99%: {:6.3f} ms"
                    .format(epoch, sum(latency_all) / len(latency_all) * 1000,
                            sum(latency_all[:int(len(latency_all) * 0.9)]) / int(len(latency_all) * 0.9) * 1000,
                            sum(latency_all[:int(len(latency_all) * 0.95)]) / int(len(latency_all) * 0.95) * 1000,
                            sum(latency_all[:int(len(latency_all) * 0.99)]) / int(len(latency_all) * 0.99) * 1000,
                            ))
                dllogger.log(step=tuple(),
                             data={"inference_sequences_per_second": float(infer_perf_avg.result().numpy()), 
                                   "e2e_inference_time": e2e_infer_time})

    if is_main_process() and args.do_train and args.do_eval:
        print(
            "**RESULTS SUMMARY** - EM: {:6.3f}, F1: {:6.3f}, Train_Time: {:4.0f} s, Train_Perf: {:4.0f} seq/s, Infer_Perf: {:4.0f} seq/s"
            .format(exact_match, f1, total_train_time, epoch_perf_avg.result() * get_world_size(),
                    infer_perf_avg.result()))
        dllogger.log(step=tuple(), data={"exact_match": exact_match, "F1": f1})


if __name__ == "__main__":
    main()
