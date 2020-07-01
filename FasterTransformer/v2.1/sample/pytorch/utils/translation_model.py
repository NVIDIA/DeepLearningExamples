# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters
import onmt.modules
from onmt.encoders.transformer import TransformerEncoder

from onmt.modules import Embeddings, VecEmbedding, CopyGenerator
from onmt.modules.util_class import Cast
from onmt.utils.misc import use_gpu
from onmt.utils.parse import ArgumentParser

from .decoding import FTDecoderLayer, DecodingWeights, CustomDecoding, TorchDecoding, TransformerDecoder


def build_embeddings(opt, text_field, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

    if opt.model_type == "vec" and for_encoder:
        return VecEmbedding(
            opt.feat_vec_size,
            emb_dim,
            position_encoding=opt.position_encoding,
            dropout=(opt.dropout[0] if type(opt.dropout) is list
                     else opt.dropout),
        )

    pad_indices = [f.vocab.stoi[f.pad_token] for _, f in text_field]
    word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]

    num_embs = [len(f.vocab) for _, f in text_field]
    num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

    fix_word_vecs = opt.fix_word_vecs_enc if for_encoder \
        else opt.fix_word_vecs_dec

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feat_pad_indices,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam",
        fix_word_vecs=fix_word_vecs
    )
    return emb


def load_test_model(opt, args):
    model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocab = checkpoint['vocab']
    if inputters.old_style_vocab(vocab):
        fields = inputters.load_old_vocab(
            vocab, opt.data_type, dynamic_dict=model_opt.copy_attn
        )
    else:
        fields = vocab

    model = build_base_model(model_opt, fields, use_gpu(opt), args, checkpoint,
                             opt.gpu)
    if args.data_type == 'fp32':
        model.float()
    elif args.data_type == 'fp16':
        model.half()
    else:
        raise ValueError('wrong data_type argument {}'.format(args.data_type))
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def build_base_model(model_opt, fields, gpu, args, checkpoint=None, gpu_id=None):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        fields (dict[str, torchtext.data.Field]):
            `Field` objects for the model.
        gpu (bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
        gpu_id (int or NoneType): Which GPU to use.

    Returns:
        the NMTModel.
    """

    # for back compat when attention_dropout was not defined
    try:
        model_opt.attention_dropout
    except AttributeError:
        model_opt.attention_dropout = model_opt.dropout

    # Build embeddings.
    if model_opt.model_type == "text" or model_opt.model_type == "vec":
        src_field = fields["src"]
        src_emb = build_embeddings(model_opt, src_field)
    else:
        src_emb = None

    # Build encoder.
    encoder = TransformerEncoder.from_opt(model_opt, src_emb)

    # Build decoder.
    tgt_field = fields["tgt"]
    tgt_emb = build_embeddings(model_opt, tgt_field, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        assert src_field.base_field.vocab == tgt_field.base_field.vocab, \
            "preprocess with -share_vocab if you use share_embeddings"

        tgt_emb.word_lut.weight = src_emb.word_lut.weight

    decoder = TransformerDecoder.from_opt(model_opt, tgt_emb, args)

    # Build NMTModel(= encoder + decoder).
    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")
    model = onmt.models.NMTModel(encoder, decoder)

    # Build Generator.
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size,
                      len(fields["tgt"].base_field.vocab)),
            Cast(torch.float32),
            gen_func
        )
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        tgt_base_field = fields["tgt"].base_field
        vocab_size = len(tgt_base_field.vocab)
        pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
        generator = CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)
        if model_opt.share_decoder_embeddings:
            generator.linear.weight = decoder.embeddings.word_lut.weight

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s

        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}
        # end of patch for backward compatibility

        model.load_state_dict(checkpoint['model'], strict=False)
        generator.load_state_dict(checkpoint['generator'], strict=False)

        if args.model_type == 'decoder_ext':
            w = []
            for i in range(model_opt.dec_layers):
                w.append([
                    decoder.transformer_layers[i].layer_norm_1.weight.data,
                    decoder.transformer_layers[i].layer_norm_1.bias.data,
                    decoder.transformer_layers[i].self_attn.linear_query.weight.data.transpose(-1, -2).contiguous(),
                    decoder.transformer_layers[i].self_attn.linear_keys.weight.data.transpose(-1, -2).contiguous(),
                    decoder.transformer_layers[i].self_attn.linear_values.weight.data.transpose(-1, -2).contiguous(),
                    decoder.transformer_layers[i].self_attn.linear_query.bias.data,
                    decoder.transformer_layers[i].self_attn.linear_keys.bias.data,
                    decoder.transformer_layers[i].self_attn.linear_values.bias.data,
                    decoder.transformer_layers[i].self_attn.final_linear.weight.data.transpose(-1, -2).contiguous(),
                    decoder.transformer_layers[i].self_attn.final_linear.bias.data,
                    decoder.transformer_layers[i].layer_norm_2.weight.data,
                    decoder.transformer_layers[i].layer_norm_2.bias.data,
                    decoder.transformer_layers[i].context_attn.linear_query.weight.data.transpose(-1, -2).contiguous(),
                    decoder.transformer_layers[i].context_attn.linear_keys.weight.data.transpose(-1, -2).contiguous(),
                    decoder.transformer_layers[i].context_attn.linear_values.weight.data.transpose(-1, -2).contiguous(),
                    decoder.transformer_layers[i].context_attn.linear_query.bias.data,
                    decoder.transformer_layers[i].context_attn.linear_keys.bias.data,
                    decoder.transformer_layers[i].context_attn.linear_values.bias.data,
                    decoder.transformer_layers[i].context_attn.final_linear.weight.data.transpose(-1, -2).contiguous(),
                    decoder.transformer_layers[i].context_attn.final_linear.bias.data,
                    decoder.transformer_layers[i].feed_forward.layer_norm.weight.data,
                    decoder.transformer_layers[i].feed_forward.layer_norm.bias.data,
                    decoder.transformer_layers[i].feed_forward.w_1.weight.data.transpose(-1, -2).contiguous(),
                    decoder.transformer_layers[i].feed_forward.w_1.bias.data,
                    decoder.transformer_layers[i].feed_forward.w_2.weight.data.transpose(-1, -2).contiguous(),
                    decoder.transformer_layers[i].feed_forward.w_2.bias.data
                ])
                for i in range(len(w[-1])):
                    w[-1][i] = w[-1][i].cuda()
                if args.data_type == 'fp16':
                    for i in range(len(w[-1])):
                        w[-1][i] = w[-1][i].half()
            decoder_layers = nn.ModuleList(
                [FTDecoderLayer(model_opt.heads, model_opt.dec_rnn_size // model_opt.heads, w[i], args) for i in range(model_opt.dec_layers)])
            model.decoder.transformer_layers = decoder_layers
        elif args.model_type == 'decoding_ext':
            vocab_size = len(fields["tgt"].base_field.vocab)
            bos_idx = fields["tgt"].base_field.vocab.stoi[fields["tgt"].base_field.init_token]
            eos_idx = fields["tgt"].base_field.vocab.stoi[fields["tgt"].base_field.eos_token]
            decoding_weights = DecodingWeights(model_opt.dec_layers, model_opt.dec_rnn_size, vocab_size, checkpoint)
            decoding_weights.to_cuda()
            if args.data_type == 'fp16':
                decoding_weights.to_half()
            model.decoder = CustomDecoding(model_opt.dec_layers, model_opt.heads, model_opt.dec_rnn_size // model_opt.heads,
                                            vocab_size, bos_idx, eos_idx, decoding_weights, args=args)
        elif args.model_type == 'torch_decoding' or args.model_type == 'torch_decoding_with_decoder_ext':
            vocab_size = len(fields["tgt"].base_field.vocab)
            bos_idx = fields["tgt"].base_field.vocab.stoi[fields["tgt"].base_field.init_token]
            eos_idx = fields["tgt"].base_field.vocab.stoi[fields["tgt"].base_field.eos_token]
            decoding_weights = DecodingWeights(model_opt.dec_layers, model_opt.dec_rnn_size, vocab_size, checkpoint)
            decoding_weights.to_cuda()
            if args.data_type == 'fp16':
                decoding_weights.to_half()
            model.decoder = TorchDecoding(model_opt.dec_layers, model_opt.heads, model_opt.dec_rnn_size // model_opt.heads,
                                            vocab_size, bos_idx, eos_idx, decoding_weights, args=args)

    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec)

    model.generator = generator
    model.to(device)
    if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
        model.half()
    return model
