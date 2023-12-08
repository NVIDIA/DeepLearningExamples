# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import os
import sys


def populate(parser):
    choices = ["pretrain", "finetune"]
    parser.add_argument("mode", help="Training mode", choices=choices)
    mode = parser.parse_args([a for a in sys.argv[1:] if a in choices]).mode

    if mode == "pretrain":
        populate_pretraining(parser)
    else:
        populate_finetuning(parser)

    populate_common(parser)
    return parser


def populate_infer(parser):
    populate_finetuning(parser)
    populate_common(parser)
    _populate_infer(parser)
    return parser


def populate_common(parser):
    train = parser.add_argument_group("training setup")
    train.add_argument("--epochs_this_job", default=0, type=int,
                       help="Run for a number of epochs and exit")
    train.add_argument("--cudnn_benchmark", action="store_true",
                       help="Enable cudnn benchmark")
    train.add_argument("--local_rank", "--local-rank", default=os.getenv("LOCAL_RANK", 0),
                       type=int, help="GPU id used for distributed training")

    optim = parser.add_argument_group("optimization setup")
    optim.add_argument("--optimizer", default="adam", type=str,
                       help="Optimization algorithm")
    optim.add_argument("--ema", type=float, default=0.0,
                       help="Discount factor for EMA of model weights")

    io = parser.add_argument_group("feature and checkpointing setup")
    io.add_argument("--log_frequency", default=1, type=int,
                    help="Number of steps between printing training stats")
    io.add_argument("--output_dir", type=str, required=True,
                    help="Directory for logs and checkpoints")
    io.add_argument("--log_file", type=str, default=None,
                    help="Path to save the training logfile.")
    io.add_argument("--benchmark_epochs_num", type=int, default=3,
                    help="Number of last epochs to calculate throughput stats")

    ckpt = parser.add_argument_group("checkpoint")
    ckpt.add_argument("--no_save", action="store_true",
                      help="Don't save models or checkpoints")
    ckpt.add_argument("--resume", action="store_true",
                      help="Try to resume from last saved checkpoint")
    ckpt.add_argument("--ckpt", default=None, type=str,
                      help="Path to a checkpoint for resuming training")
    ckpt.add_argument("--save_frequency", default=10, type=int,
                      help="Checkpoint saving frequency in epochs")
    ckpt.add_argument("--keep_milestones", default=[100, 200, 300, 400],
                      type=int, nargs="+",
                      help="Milestone checkpoints to keep from removing")
    # io.add_argument("--save_best_from", default=380, type=int,
    #                 help="Epoch on which to begin tracking best checkpoint (dev WER)")

    common = parser.add_argument_group("common")
    common.add_argument("--seed", type=int, default=1,
                        help="Pseudo random number generator seed")
    common.add_argument("--cpu", action="store_true",
                        help="Use CPU instead of CUDA")
    common.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")
    common.add_argument("--fp16", action="store_true",
                        help="If fp16 is being used")
    common.add_argument("--bf16", action="store_true",
                        help="Train in bfloat16 precision")
    common.add_argument("--min_loss_scale", type=float, default=0.0001,
                        help="Minimum FP16/AMP loss scale, after which "
                             "training is stopped")
    common.add_argument("--fp16_init_scale", type=int, default=128,
                        help="Default FP16 loss scale")

    common.add_argument("--fp32_transformer_layernorm", action="store_true",
                        help="Calculate MHA LayerNorms in full precision")
    common.add_argument("--fp32_mha_softmax", action="store_true",
                        help="Calculate multi-head attention to FP32")
    common.add_argument("--fp32_cosine_sim", action="store_true",
                        help="Calculate cosine similarity in FP32")
    common.add_argument("--fp32_pos_conv", action="store_true",
                        help="Calculate positional conv in FP32")
    common.add_argument("--fp32_conv_norms", action="store_true",
                        help="Calculate normalization in conv layers in FP32")

    common.add_argument("--mha", type=str, default="fairseq",
                        choices=["fairseq", "pyt"], help="MHA implementation")

    common.add_argument("--num_concat_batches", type=int, default=1)

    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--num_workers", type=int, default=6,
                         help="How many subprocesses to use for data loading")
    dataset.add_argument("--skip_invalid_size_inputs_valid_test",
                         action="store_true",
                         help="Ignore too long or too short lines in valid and"
                              " test set")
    dataset.add_argument("--max_tokens", type=int, default=1400000,
                         help="Maximum number of tokens in a batch")
    dataset.add_argument("--max_tokens_valid", type=int, default=1400000,
                         help="Maximum number of tokens in a validation batch "
                              "(defaults to --max-tokens)")
    dataset.add_argument("--required_batch_size_multiple", type=int, default=8,
                         help="Batch size will be a multiplier of this value")
    dataset.add_argument("--required_seq_len_multiple", type=int, default=2,
                         help="Pad the input to encoder such that the sequence"
                              " length is divisible by multiple")
    dataset.add_argument("--train_subset", type=str, default="train",
                         help="Data subset to use for training (e.g. train, "
                              "valid, test)")
    dataset.add_argument("--valid_subset", type=str, default="valid",
                         help="Comma separated list of data subsets to use for"
                              " validation (e.g. train, valid, test)")
    dataset.add_argument("--batch_size", type=int, default=None,
                         help="Number of examples in a batch")
    dataset.add_argument("--batch_size_valid", type=int, default=None,
                         help="Batch size of the validation batch (defaults "
                              "to --batch-size)")

    task = parser.add_argument_group("task")
    task.add_argument("--data", type=str,
                      default="/workspace/fairseq/librispeech",
                      help="Path to data directory")
    task.add_argument("--sample_rate", type=int, default=16000,
                      help="Target sample rate. audio files will be up/down "
                           "sampled to this rate")
    task.add_argument("--enable_padding", action="store_true",
                      help="Pad shorter samples instead of cropping")
    task.add_argument("--min_sample_size", type=int, default=None,
                      help="Min sample size to crop to for batching")
    task.add_argument("--max_sample_size", type=int, default=None,
                      help="Max sample size to crop to for batching")
    task.add_argument("--num_batch_buckets", type=int, default=0,
                      help="If >0, then bucket source and target lengths into "
                           "N buckets and pad accordingly; this is useful on "
                           "TPUs to minimize the number of compilations")

    opt = parser.add_argument_group("optimization & optimizer")
    opt.add_argument("--max_update", type=int, default=400000,
                     help="Force stop training at specified update")
    opt.add_argument("--update_freq", type=int, nargs="+", default=[64],
                     help="Accumulate grads and update params every N batches")
    opt.add_argument("--lr", type=float, nargs="+", default=[0.0005],
                     help="Max learning rate, must be more than cfg.min_lr")
    opt.add_argument("--adam_betas", type=float, nargs="+", default=[0.9, 0.98],
                     help="Betas for Adam optimizer")
    opt.add_argument("--adam_eps", type=float, default=1e-06,
                     help="Epsilon for Adam optimizer")
    opt.add_argument("--weight_decay", type=float, default=0.01,
                     help="Weight decay")
    opt.add_argument("--clip_norm", type=float, default=0.0,
                     help="Clip threshold of gradients")

    sched = parser.add_argument_group("lr_scheduler")
    sched.add_argument("--lr_policy", type=str, default="poly",
                       choices=["poly", "exp"], help="LR decay policy")
    sched.add_argument("--warmup_updates", type=int, default=32000,
                       help="Warmup the learning rate linearly for the first "
                            "N updates")
    sched.add_argument("--hold_updates", type=int, default=0,
                       help="The number of updates with const learning rate")
    sched.add_argument("--initial_lr_scale", type=float, default=0.0,
                       help="Initial learning rate scale")
    sched.add_argument("--final_lr_scale", type=float, default=0.0,
                       help="Final learning rate scale")
    sched.add_argument("--lr_poly_power", type=float, default=1.0,
                       help="Poly lr policy policy power")
    sched.add_argument("--lr_exp_decay", type=float, default=None,
                       help="Exp lr policy decay factor")

    drop = parser.add_argument_group("dropout")
    drop.add_argument("--dropout", type=float, default=0.1,
                      help="Dropout probability for the transformer")
    drop.add_argument("--attention_dropout", type=float, default=0.0,
                      help="Dropout probability for attention weights")
    drop.add_argument("--activation_dropout", type=float, default=0.0,
                      help="Dropout probability after activation in FFN")
    drop.add_argument("--dropout_input", type=float, default=0.1,
                      help="Dropout to apply to the input (after feat extr)")
    drop.add_argument("--dropout_features", type=float, default=0.1,
                      help="Dropout to apply to the features (after feat extr)")

    mask = parser.add_argument_group("input masking")
    mask.add_argument("--apply_mask", action="store_true",
                      help="Apply masking during fine-tuning")
    mask.add_argument("--mask_length", type=int, default=10,
                      help="Repeat the mask indices multiple times")
    mask.add_argument("--mask_prob", type=float, default=0.5,
                      help="Probability of replacing a token with mask "
                           "(normalized by length)")
    mask.add_argument("--require_same_masks", type=bool, default=True,
                      help="Whether to number of masked timesteps must be the"
                           " same across all examples in a batch")
    mask.add_argument("--mask_selection", default="static",
                      choices=["static", "uniform", "normal", "poisson"],
                      help="How to choose masks")
    mask.add_argument("--mask_other", type=float, default=0,
                      help="Secondary mask argument (used for more complex "
                           "distributions), see help in compute_mask_indices")
    mask.add_argument("--no_mask_overlap", type=bool, default=False,
                      help="Whether to allow masks to overlap")
    mask.add_argument("--mask_min_space", type=int, default=1,
                      help="Min space between spans (if no overlap is enabled)")
    mask.add_argument("--mask_channel_length", type=int, default=10,
                      help="Length of the mask for features (channels)")
    mask.add_argument("--mask_channel_prob", type=float, default=0.0,
                      help="Probability of replacing a feature with 0")
    mask.add_argument("--mask_channel_before", type=bool, default=False,
                      help="Apply channel-masking before frequency-masking")
    mask.add_argument("--mask_channel_selection", default="static",
                      choices=["static", "uniform", "normal", "poisson"],
                      help="How to choose mask length for channel masking")
    mask.add_argument("--mask_channel_other", type=float, default=0,
                      help="Secondary mask argument (used for more complex "
                           "distributions), see help in compute_mask_indicesh")
    mask.add_argument("--no_mask_channel_overlap", type=bool, default=False,
                      help="Whether to allow channel masks to overlap")
    mask.add_argument("--mask_channel_min_space", type=int, default=1,
                      help="Min space between spans (if no overlap is enabled)")
    parser.add_argument("--feature_grad_mult", type=float, default=0.1,
                        help="Reset feature grad mult in wav2vec 2.0 to this")
    # NOTE In Fairseq this is called `--layerdrop` in fine-tuning yamls
    parser.add_argument("--encoder_layerdrop", type=float, default=0.05,
                        help="Probability of dropping a layer in wav2vec 2.0")
    mask.add_argument("--mask_dropout", type=float, default=0.0,
                      help="Percent of masks to unmask for each sample")


def populate_finetuning(parser):
    """Args for fine-tuning, absent from pre-trained ckpts."""
    ft = parser.add_argument_group("supervised fine-tuning")
    ft.add_argument("--final_dropout", type=float, default=0.0,
                    help="Dropout after transformer and before final proj")
    ft.add_argument("--w2v_path", type=str, default=None,
                    help="Path to wav2vec 2.0 model")
    ft.add_argument("--blank_weight", type=float, default=0)
    ft.add_argument("--blank_mode", type=str, default="add")
    ft.add_argument("--labels", type=str, default="ltr",
                    help="Extension of the label file to load for fine-tuning")
    ft.add_argument("--freeze_finetune_updates", type=int, default=0,
                    help="Don't finetune wav2vec for this many updates")


def populate_pretraining(parser):
    """During fine-tuning these parameters will be loaded from a ckpt."""
    model = parser.add_argument_group("model")
    model.add_argument("--extractor_mode", type=str, default="default",
                       help="Mode for feature extractor. default has a single "
                            "group norm with d groups in the first conv block,"
                            " whereas layer_norm has layer norms in every "
                            "block (meant to use with normalize=True)")
    model.add_argument("--encoder_layers", type=int, default=12,
                      help="Num encoder layers in the transformer")
    model.add_argument("--encoder_embed_dim", type=int, default=768,
                      help="Encoder embedding dimension")
    model.add_argument("--encoder_ffn_embed_dim", type=int, default=3072,
                      help="Encoder embedding dimension for FFN")
    model.add_argument("--encoder_attention_heads", type=int, default=12,
                      help="Num encoder attention heads")
    model.add_argument("--activation_fn", type=str, default="gelu",
                      help="Activation function to use")
    model.add_argument("--final_dim", type=int, default=256,
                       help="Project final representations and targets to this"
                            " many dimensions. set to encoder_embed_dim "
                            "is <= 0")
    model.add_argument("--layer_norm_first", action="store_true",
                       help="Apply layernorm first in the transformer")
    model.add_argument("--conv_feature_layers", type=str,
                       default="[(512,10,5)]+[(512,3,2)]*4+[(512,2,2)]+[(512,2,2)]",
                       help="String describing convolutional feature "
                            "extraction layers in form of a python list that "
                            "contains [(dim, kernel_size, stride), ...]")
    model.add_argument("--conv_bias", action="store_true",
                       help="Include bias in conv encoder")
    model.add_argument("--logit_temp", type=float, default=0.1,
                       help="Temperature to divide logits by")
    model.add_argument("--quantize_targets", action="store_true",
                       help="Use quantized targets")
    model.add_argument("--quantize_input", action="store_true",
                       help="Use quantized inputs")
    model.add_argument("--target_glu", action="store_true",
                       help="Adds projection + glu to targets")
    model.add_argument("--quantizer_depth", type=int, default=1,
                       help="Number of quantizer layers")
    model.add_argument("--quantizer_factor", type=int, default=3,
                       help="Dimensionality increase for inner quantizer "
                            "layers (if depth > 1)")
    model.add_argument("--latent_vars", type=int, default=320,
                       help="Number of latent variables V in each group of the"
                            " codebook")
    model.add_argument("--latent_groups", type=int, default=2,
                       help="Number of groups G of latent variables in the "
                            "codebook")
    model.add_argument("--latent_dim", type=int, default=0,
                       help="If > 0, uses this dimensionality for latent var"
                            "iables. otherwise uses final_dim / latent_groups")
    model.add_argument("--num_negatives", type=int, default=100,
                       help="Num of sampled negatives")
    model.add_argument("--negatives_from_everywhere", action="store_true",
                       help="Sample negatives from everywhere, not just masked"
                            " states")
    model.add_argument("--cross_sample_negatives", type=int, default=0,
                       help="Num of cross sampled negatives")
    model.add_argument("--codebook_negatives", type=int, default=0,
                       help="Number of negative examples codebook")
    model.add_argument("--conv_pos", type=int, default=128,
                       help="Number of filters for convolutional positional "
                            "embeddings")
    model.add_argument("--conv_pos_groups", type=int, default=16,
                       help="Number of groups for convolutional positional "
                            "embedding")
    model.add_argument("--latent_temp", type=float, nargs="+",
                       default=[2.0, 0.5, 0.999995],
                       help="Legacy (to be removed)")
    model.add_argument("--normalize", action="store_true",
                       help="If set, normalizes input to have 0 mean and unit "
                            "variance")
    parser.add_argument("--log_keys", type=str, nargs="*",
                        default=["prob_perplexity", "code_perplexity", "temp"],
                        help="Additional output keys to log")

    crit = parser.add_argument_group("criterion")
    crit.add_argument("--infonce", action="store_true",
                      help="If set, uses cross entropy instead of binary cross"
                           " entropy (i.e. InfoNCE loss)")
    crit.add_argument("--loss_weights", type=float, nargs="*",
                      default=[0.1, 10.0], help="Weights for the loss terms")

    joc = parser.add_argument_group("joc experimental")
    joc.add_argument("--use_spectrogram_features", action="store_true",
                     help="Train on input spectrograms")
    joc.add_argument("--rotary_embeddings", action="store_true",
                     help="Use rotarty embeddings for Transformer layers")
    joc.add_argument("--hourglass_transformer", type=str, default=None,
                     help="Specify the number of layers and shorteining, e.g.,"
                          " [n_pre,(n_hourglass, shorten_factor),n_post]")
    joc.add_argument("--hourglass_resample", type=str, default="naive",
                     help="Method of up/downsampling in the hourglass model")
    joc.add_argument("--spectrogram_feature_stacking", type=int, default=1)
    joc.add_argument("--spectrogram_feature_subsampling", type=int, default=1)
    joc.add_argument("--spectrogram_window_size", type=float, default=0.02)
    joc.add_argument("--spectrogram_window_stride", type=float, default=0.01)
    joc.add_argument("--spectrogram_n_filt", type=int, default=80)
    return parser


def _populate_infer(parser):
    # Fine-tuning only
    infer = parser.add_argument_group("inference")
    infer.add_argument("--steps", default=0, type=int,
                       help="Eval this many steps for every worker")
    infer.add_argument("--warmup_steps", default=0, type=int,
                       help="Burn-in period before measuring latencies")
    infer.add_argument("--labels_path", type=str, default=None,
                       help="Path to output labels file, e.g., dict.ltr.txt")
    infer.add_argument("--save_predictions", type=str, default=None,
                       help="Save predictions in text form at this location")
    infer.add_argument("--save_logits", default=None, type=str,
                       help="Save output logits under specified path")
    infer.add_argument("--transcribe_wav", type=str,
                       help="Path to a single .wav file (16KHz)")
    infer.add_argument("--transcribe_filelist", type=str,
                       help="Path to a filelist with one .wav path per line")
    infer.add_argument("--torchscript", action="store_true",
                       help="Evaluate with a TorchScripted model")
    infer.add_argument("--w2v_path_for_args", type=str, default=None,
                       help="Args to build model for inference (weights will "
                            "be loaded from --w2v_path)")
