# Copyright 2018 MLBenchmark Group. All Rights Reserved.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

# Common values reported

VALUE_EPOCH = "epoch"
VALUE_ITERATION = "iteration"
VALUE_ACCURACY = "accuracy"
VALUE_BLEU = "bleu"
VALUE_TOP1 = "top1"
VALUE_TOP5 = "top5"
VALUE_BBOX_MAP = "bbox_map"
VALUE_MASK_MAP = "mask_map"
VALUE_BCE = "binary_cross_entropy"


# Timed blocks (used with timed_function & timed_block
# For each there should be *_start and *_stop tags defined

RUN_BLOCK = "run"
SETUP_BLOCK = "setup"
PREPROC_BLOCK = "preproc"

TRAIN_BLOCK = "train"
TRAIN_PREPROC_BLOCK = "train_preproc"
TRAIN_EPOCH_BLOCK = "train_epoch"
TRAIN_EPOCH_PREPROC_BLOCK = "train_epoch_preproc"
TRAIN_CHECKPOINT_BLOCK = "train_checkpoint"
TRAIN_ITER_BLOCK = "train_iteration"

EVAL_BLOCK = "eval"
EVAL_ITER_BLOCK = "eval_iteration"

TIMED_BLOCKS = {
    RUN_BLOCK,
    SETUP_BLOCK,
    PREPROC_BLOCK,
    TRAIN_BLOCK,
    TRAIN_PREPROC_BLOCK,
    TRAIN_EPOCH_BLOCK,
    TRAIN_EPOCH_PREPROC_BLOCK,
    TRAIN_CHECKPOINT_BLOCK,
    TRAIN_ITER_BLOCK,
    EVAL_BLOCK,
    EVAL_ITER_BLOCK,
}


# Events

RUN_INIT = "run_init"

SETUP_START = "setup_start"
SETUP_STOP = "setup_stop"

PREPROC_START = "preproc_start"
PREPROC_STOP = "preproc_stop"

RUN_START = "run_start"
RUN_STOP = "run_stop"
RUN_FINAL = "run_final"

TRAIN_CHECKPOINT_START = "train_checkpoint_start"
TRAIN_CHECKPOINT_STOP = "train_checkpoint_stop"

TRAIN_PREPROC_START = "train_preproc_start"
TRAIN_PREPROC_STOP = "train_preproc_stop"

TRAIN_EPOCH_PREPROC_START = "train_epoch_preproc_start"
TRAIN_EPOCH_PREPROC_STOP = "train_epoch_preproc_stop"

TRAIN_ITER_START = "train_iter_start"
TRAIN_ITER_STOP = "train_iter_stop"

TRAIN_EPOCH_START = "train_epoch_start"
TRAIN_EPOCH_STOP = "train_epoch_stop"


# MLPerf specific tags

RUN_CLEAR_CACHES = "run_clear_caches"

PREPROC_NUM_TRAIN_EXAMPLES = "preproc_num_train_examples"
PREPROC_NUM_EVAL_EXAMPLES = "preproc_num_eval_examples"
PREPROC_TOKENIZE_TRAINING = "preproc_tokenize_training"
PREPROC_TOKENIZE_EVAL = "preproc_tokenize_eval"
PREPROC_VOCAB_SIZE = "preproc_vocab_size"

RUN_SET_RANDOM_SEED = "run_set_random_seed"

INPUT_SIZE = "input_size"
INPUT_BATCH_SIZE = "input_batch_size"
INPUT_ORDER = "input_order"
INPUT_SHARD = "input_shard"
INPUT_BN_SPAN = "input_bn_span"

INPUT_CENTRAL_CROP = "input_central_crop"
INPUT_CROP_USES_BBOXES = "input_crop_uses_bboxes"
INPUT_DISTORTED_CROP_MIN_OBJ_COV = "input_distorted_crop_min_object_covered"
INPUT_DISTORTED_CROP_RATIO_RANGE = "input_distorted_crop_aspect_ratio_range"
INPUT_DISTORTED_CROP_AREA_RANGE = "input_distorted_crop_area_range"
INPUT_DISTORTED_CROP_MAX_ATTEMPTS = "input_distorted_crop_max_attempts"
INPUT_MEAN_SUBTRACTION = "input_mean_subtraction"
INPUT_RANDOM_FLIP = "input_random_flip"

INPUT_RESIZE = "input_resize"
INPUT_RESIZE_ASPECT_PRESERVING = "input_resize_aspect_preserving"


# Opt

OPT_NAME = "opt_name"

OPT_LR = "opt_learning_rate"
OPT_MOMENTUM = "opt_momentum"

OPT_WEIGHT_DECAY = "opt_weight_decay"

OPT_HP_ADAM_BETA1 = "opt_hp_Adam_beta1"
OPT_HP_ADAM_BETA2 = "opt_hp_Adam_beta2"
OPT_HP_ADAM_EPSILON = "opt_hp_Adam_epsilon"

OPT_LR_WARMUP_STEPS = "opt_learning_rate_warmup_steps"


#  Train

TRAIN_LOOP = "train_loop"
TRAIN_EPOCH = "train_epoch"
TRAIN_CHECKPOINT = "train_checkpoint"
TRAIN_LOSS = "train_loss"
TRAIN_ITERATION_LOSS = "train_iteration_loss"


# Eval

EVAL_START = "eval_start"
EVAL_SIZE = "eval_size"
EVAL_TARGET = "eval_target"
EVAL_ACCURACY = "eval_accuracy"
EVAL_STOP = "eval_stop"


# Perf

PERF_IT_PER_SEC = "perf_it_per_sec"
PERF_TIME_TO_TRAIN = "time_to_train"

EVAL_ITERATION_ACCURACY = "eval_iteration_accuracy"


# Model

MODEL_HP_LOSS_FN = "model_hp_loss_fn"

MODEL_HP_INITIAL_SHAPE = "model_hp_initial_shape"
MODEL_HP_FINAL_SHAPE = "model_hp_final_shape"

MODEL_L2_REGULARIZATION = "model_l2_regularization"
MODEL_EXCLUDE_BN_FROM_L2 = "model_exclude_bn_from_l2"

MODEL_HP_RELU = "model_hp_relu"
MODEL_HP_CONV2D_FIXED_PADDING = "model_hp_conv2d_fixed_padding"
MODEL_HP_BATCH_NORM = "model_hp_batch_norm"
MODEL_HP_DENSE = "model_hp_dense"


# GNMT specific

MODEL_HP_LOSS_SMOOTHING = "model_hp_loss_smoothing"
MODEL_HP_NUM_LAYERS = "model_hp_num_layers"
MODEL_HP_HIDDEN_SIZE = "model_hp_hidden_size"
MODEL_HP_DROPOUT = "model_hp_dropout"

EVAL_HP_BEAM_SIZE = "eval_hp_beam_size"
TRAIN_HP_MAX_SEQ_LEN = "train_hp_max_sequence_length"
EVAL_HP_MAX_SEQ_LEN = "eval_hp_max_sequence_length"
EVAL_HP_LEN_NORM_CONST = "eval_hp_length_normalization_constant"
EVAL_HP_LEN_NORM_FACTOR = "eval_hp_length_normalization_factor"
EVAL_HP_COV_PENALTY_FACTOR = "eval_hp_coverage_penalty_factor"


# NCF specific

PREPROC_HP_MIN_RATINGS = "preproc_hp_min_ratings"
PREPROC_HP_NUM_EVAL = "preproc_hp_num_eval"
PREPROC_HP_SAMPLE_EVAL_REPLACEMENT = "preproc_hp_sample_eval_replacement"

INPUT_HP_NUM_NEG = "input_hp_num_neg"
INPUT_HP_SAMPLE_TRAIN_REPLACEMENT = "input_hp_sample_train_replacement"
INPUT_STEP_TRAIN_NEG_GEN = "input_step_train_neg_gen"
INPUT_STEP_EVAL_NEG_GEN = "input_step_eval_neg_gen"

EVAL_HP_NUM_USERS = "eval_hp_num_users"
EVAL_HP_NUM_NEG = "eval_hp_num_neg"

MODEL_HP_MF_DIM = "model_hp_mf_dim"
MODEL_HP_MLP_LAYER_SIZES = "model_hp_mlp_layer_sizes"


# RESNET specific

EVAL_EPOCH_OFFSET = "eval_offset"

MODEL_HP_INITIAL_MAX_POOL = "model_hp_initial_max_pool"
MODEL_HP_BEGIN_BLOCK = "model_hp_begin_block"
MODEL_HP_END_BLOCK = "model_hp_end_block"
MODEL_HP_BLOCK_TYPE = "model_hp_block_type"
MODEL_HP_PROJECTION_SHORTCUT = "model_hp_projection_shortcut"
MODEL_HP_SHORTCUT_ADD = "model_hp_shorcut_add"
MODEL_HP_RESNET_TOPOLOGY = "model_hp_resnet_topology"


# Transformer specific

INPUT_MAX_LENGTH = "input_max_length"

MODEL_HP_INITIALIZER_GAIN = "model_hp_initializer_gain"
MODEL_HP_VOCAB_SIZE = "model_hp_vocab_size"
MODEL_HP_NUM_HIDDEN_LAYERS = "model_hp_hidden_layers"
MODEL_HP_EMBEDDING_SHARED_WEIGHTS = "model_hp_embedding_shared_weights"
MODEL_HP_ATTENTION_DENSE = "model_hp_attention_dense"
MODEL_HP_ATTENTION_DROPOUT = "model_hp_attention_dropout"
MODEL_HP_FFN_OUTPUT_DENSE = "model_hp_ffn_output_dense"
MODEL_HP_FFN_FILTER_DENSE = "model_hp_ffn_filter_dense"
MODEL_HP_RELU_DROPOUT = "model_hp_relu_dropout"
MODEL_HP_LAYER_POSTPROCESS_DROPOUT = "model_hp_layer_postprocess_dropout"
MODEL_HP_NORM = "model_hp_norm"
MODEL_HP_SEQ_BEAM_SEARCH = "model_hp_sequence_beam_search"

