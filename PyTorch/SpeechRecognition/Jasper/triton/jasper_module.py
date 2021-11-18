# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import torch
import sys
sys.path.append("./")

class FeatureCollate:

    def __init__(self, feature_proc):
        self.feature_proc = feature_proc

    def __call__(self, batch):
        bs = len(batch)
        max_len = lambda l,idx: max(el[idx].size(0) for el in l)
        audio = torch.zeros(bs, max_len(batch, 0))
        audio_lens = torch.zeros(bs, dtype=torch.int32)

        for i, sample in enumerate(batch):
            audio[i].narrow(0, 0, sample[0].size(0)).copy_(sample[0])
            audio_lens[i] = sample[1]

        ret = (audio, audio_lens)

        if self.feature_proc is not None:
            feats, feat_lens = self.feature_proc(audio, audio_lens)
            ret = (feats,)

        return ret


def get_dataloader(model_args_list):
    ''' return dataloader for inference '''

    from inference import get_parser
    from common.helpers import add_ctc_blank
    from jasper import config
    from common.dataset import (AudioDataset, FilelistDataset, get_data_loader,
                                SingleAudioDataset)
    from common.features import FilterbankFeatures

    parser = get_parser()
    parser.add_argument('--component', type=str, default="model",
                        choices=["feature-extractor", "model", "decoder"],
                        help='Component to convert')
    args = parser.parse_args(model_args_list)


    if args.component == "decoder":
        return None

    cfg = config.load(args.model_config)
    config.apply_config_overrides(cfg, args)

    symbols = add_ctc_blank(cfg['labels'])

    dataset_kw, features_kw = config.input(cfg, 'val')

    dataset = AudioDataset(args.dataset_dir, args.val_manifests,
                           symbols, **dataset_kw)

    data_loader = get_data_loader(dataset, args.batch_size, multi_gpu=False,
                                  shuffle=False, num_workers=4, drop_last=False)
    feature_proc = None

    if args.component == "model":
        feature_proc = FilterbankFeatures(**features_kw)

    data_loader.collate_fn = FeatureCollate(feature_proc)

    return data_loader


def init_feature_extractor(args):

    from jasper import config
    from common.features import FilterbankFeatures

    cfg = config.load(args.model_config)
    config.apply_config_overrides(cfg, args)
    _, features_kw = config.input(cfg, 'val')

    feature_proc = FilterbankFeatures(**features_kw)

    return feature_proc


def init_acoustic_model(args):

    from common.helpers import add_ctc_blank
    from jasper.model import Jasper
    from jasper import config

    cfg = config.load(args.model_config)
    config.apply_config_overrides(cfg, args)

    if cfg['jasper']['encoder']['use_conv_masks'] == True:
        print("[Jasper module]: Warning: setting 'use_conv_masks' \
to False; masked convolutions are not supported.")
        cfg['jasper']['encoder']['use_conv_masks'] = False

    symbols = add_ctc_blank(cfg['labels'])
    model = Jasper(encoder_kw=config.encoder(cfg),
                   decoder_kw=config.decoder(cfg, n_classes=len(symbols)))

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        key = 'ema_state_dict' if args.ema else 'state_dict'
        state_dict = checkpoint[key]
        model.load_state_dict(state_dict, strict=True)

    return model


def init_decoder(args):

    class GreedyCTCDecoderSimple(torch.nn.Module):
        @torch.no_grad()
        def forward(self, log_probs):
            return log_probs.argmax(dim=-1, keepdim=False).int()
    return GreedyCTCDecoderSimple()


def init_model(model_args_list, precision, device):
    ''' Return either of the components: feature-extractor, model, or decoder.
The returned compoenent is ready to convert '''

    from inference import get_parser
    parser = get_parser()
    parser.add_argument('--component', type=str, default="model",
                        choices=["feature-extractor", "model", "decoder"],
                        help='Component to convert')
    args = parser.parse_args(model_args_list)

    init_comp = {"feature-extractor": init_feature_extractor,
                 "model": init_acoustic_model,
                 "decoder": init_decoder}
    comp = init_comp[args.component](args)

    torch_device = torch.device(device)
    print(f"[Jasper module]: using device {torch_device}")
    comp.to(torch_device)
    comp.eval()

    if precision == "fp16":
        print("[Jasper module]: using mixed precision")
        comp.half()
    else:
        print("[Jasper module]: using fp32 precision")
    return comp
