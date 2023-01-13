# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

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

import argparse
import os
from functools import reduce
from pathlib import Path

import torch

import wav2vec2.arg_parser
from common.fairseq.data import AddTargetDataset, FileAudioDataset
from common.utils import AttrDict, print_once
from wav2vec2.model import Wav2Vec2Model, Wav2VecEncoder, Wav2VecCtc


blank_symbol = "<s>"  # for CTC


# Supervised CTC training
class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


# For frame-wise phoneme labels
class PhoneLabelEncoder:
    def __call__(self, label):
        return torch.IntTensor([int(id) for id in label.split()])


def load_dataset(split, args, target_dictionary=None, with_labels=False,
                 training=True):

    dataset = FileAudioDataset(
        manifest_path=Path(args.data, f'{split}.tsv'),
        sample_rate=args.sample_rate,
        min_sample_size=args.min_sample_size if training else None,
        max_sample_size=args.max_sample_size if training else None,
        pad=(hasattr(args, 'labels') or args.enable_padding),
        normalize=args.normalize,
        num_buckets=args.num_batch_buckets,
        compute_mask_indices=False,
        repeat_to_refsize=(args.num_concat_batches > 1),
    )

    if with_labels:
        assert args.labels
        assert hasattr(args, 'labels')

        skip_inds = getattr(dataset, "skipped_indices", set())
        with open(Path(args.data, f"{split}.{args.labels}")) as f:
            labels = [line for i, line in enumerate(f) if i not in skip_inds]

        assert len(labels) == len(dataset), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(dataset)}) do not match"
        )

        dataset = AddTargetDataset(
            dataset,
            labels,
            pad=target_dictionary.pad(),
            eos=target_dictionary.eos(),
            batch_targets=True,
            process_label=LabelEncoder(target_dictionary),
            add_to_input=False
        )

    return dataset


def load_phone_classification_dataset(split, args):

    assert not args.labels

    manifest_path = os.path.join(args.data, "{}.tsv".format(split))

    dataset = FileAudioDataset(
        manifest_path=manifest_path,
        sample_rate=args.sample_rate,
        max_sample_size=args.max_sample_size,
        min_sample_size=args.min_sample_size,
        pad=args.labels is not None or args.enable_padding,
        normalize=args.normalize,
        num_buckets=args.num_batch_buckets,
        compute_mask_indices=False,
    )

    return dataset


def _prune_infer_state_dict_prefix(state_dict,
                                   prefix='w2v_encoder.w2v_model.'):
    pref_len = len(prefix)
    return {
        (k[pref_len:] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def build_model(args, mode='pretrain', target_dictionary=None):

    cfg = AttrDict(vars(args))
    if mode == 'pretrain':
        assert target_dictionary is None
        model = Wav2Vec2Model(cfg)
    elif mode == 'finetune':
        state = torch.load(args.w2v_path, map_location='cpu')['model']
        enc = Wav2VecEncoder(cfg, state, output_size=len(target_dictionary))
        model = Wav2VecCtc(cfg, enc)
    elif mode == 'infer':
        enc = Wav2VecEncoder(cfg, None, output_size=len(target_dictionary))
        model = Wav2VecCtc(cfg, enc)
    else:
        raise ValueError

    sequence_generator = None
    tokenizer = None

    actualized_cfg = getattr(model, "cfg", None)
    if actualized_cfg is not None and "w2v_args" in actualized_cfg:
        cfg.w2v_args = actualized_cfg.w2v_args

    return model, sequence_generator, tokenizer


def build_phone_classification_model(args):
    model = Wav2VecEncoder(args)

    actualized_cfg = getattr(model, "cfg", None)
    if actualized_cfg is not None:
        if "w2v_args" in actualized_cfg:
            raise NotImplementedError
    return model


def get_ckpt_args(ckpt):
    """Return a dictionary of args saved inside a ckpt.

    Handles old and new Fairseq ckpts, Nvidia DLE ckpts.
    """
    if "cfg" in ckpt:
        import omegaconf
        w2v_args = omegaconf.OmegaConf.to_container(ckpt["cfg"])
        # Flatten nested dicts (hopefully matching keys have same values)
        w2v_args = reduce(lambda a, b: {**(a or {}), **(b or {})},
                          w2v_args.values())
    else:  # Legacy checkpoints
        w2v_args = ckpt["args"]
        if type(w2v_args) is argparse.Namespace:
            w2v_args = vars(w2v_args)
    return w2v_args


def update_args_for_finetuning(args, w2v_path_for_args):
    w2v_args = get_ckpt_args(torch.load(w2v_path_for_args, map_location="cpu"))

    pretrain_parser = argparse.ArgumentParser()
    wav2vec2.arg_parser.populate_pretraining(pretrain_parser)
    my_args = vars(pretrain_parser.parse_args([]))

    for arg in my_args:
        if arg in w2v_args and my_args[arg] != w2v_args[arg]:
            fname = Path(args.w2v_path).name
            print_once(f'Setting from {fname}: {arg}={w2v_args[arg]}',
                       local_rank=args.local_rank)
            setattr(args, arg, w2v_args[arg])
        else:
            setattr(args, arg, my_args[arg])
