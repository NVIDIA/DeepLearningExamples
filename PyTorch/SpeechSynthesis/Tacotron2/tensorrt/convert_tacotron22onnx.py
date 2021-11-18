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
from torch import nn
from torch.nn import functional as F
import argparse

import sys
sys.path.append('./')

import models
from inference import checkpoint_from_distributed, unwrap_distributed, load_and_setup_model, prepare_input_sequence
from tacotron2_common.utils import to_gpu, get_mask_from_lengths

def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('--tacotron2', type=str,
                        help='full path to the Tacotron2 model checkpoint file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory for the exported Tacotron 2 ONNX model')
    parser.add_argument('--fp16', action='store_true',
                        help='Export with half precision to ONNX')

    return parser


def encoder_infer(self, x, input_lengths):
    device = x.device
    for conv in self.convolutions:
        x = F.dropout(F.relu(conv(x.to(device))), 0.5, False)

    x = x.transpose(1, 2)

    x = nn.utils.rnn.pack_padded_sequence(
        x, input_lengths, batch_first=True)

    outputs, _ = self.lstm(x)

    outputs, _ = nn.utils.rnn.pad_packed_sequence(
        outputs, batch_first=True)

    lens = input_lengths*2

    return outputs, lens


class Encoder(torch.nn.Module):
    def __init__(self, tacotron2):
        super(Encoder, self).__init__()
        self.tacotron2 = tacotron2
        self.tacotron2.encoder.lstm.flatten_parameters()
        self.infer = encoder_infer

    def forward(self, sequence, sequence_lengths):
        embedded_inputs = self.tacotron2.embedding(sequence).transpose(1, 2)
        memory, lens = self.infer(self.tacotron2.encoder, embedded_inputs, sequence_lengths)
        processed_memory = self.tacotron2.decoder.attention_layer.memory_layer(memory)
        return memory, processed_memory, lens

class Postnet(torch.nn.Module):
    def __init__(self, tacotron2):
        super(Postnet, self).__init__()
        self.tacotron2 = tacotron2

    def forward(self, mel_outputs):
        mel_outputs_postnet = self.tacotron2.postnet(mel_outputs)
        return mel_outputs + mel_outputs_postnet

def lstmcell2lstm_params(lstm_mod, lstmcell_mod):
    lstm_mod.weight_ih_l0 = torch.nn.Parameter(lstmcell_mod.weight_ih)
    lstm_mod.weight_hh_l0 = torch.nn.Parameter(lstmcell_mod.weight_hh)
    lstm_mod.bias_ih_l0 = torch.nn.Parameter(lstmcell_mod.bias_ih)
    lstm_mod.bias_hh_l0 = torch.nn.Parameter(lstmcell_mod.bias_hh)


def prenet_infer(self, x):
    x1 = x[:]
    for linear in self.layers:
        x1 = F.relu(linear(x1))
        x0 = x1[0].unsqueeze(0)
        mask = torch.le(torch.rand(256, device='cuda').to(x.dtype), 0.5).to(x.dtype)
        mask = mask.expand(x1.size(0), x1.size(1))
        x1 = x1*mask*2.0

    return x1

class DecoderIter(torch.nn.Module):
    def __init__(self, tacotron2):
        super(DecoderIter, self).__init__()

        self.tacotron2 = tacotron2
        dec = tacotron2.decoder

        self.p_attention_dropout = dec.p_attention_dropout
        self.p_decoder_dropout = dec.p_decoder_dropout
        self.prenet = dec.prenet

        self.prenet.infer = prenet_infer

        self.attention_rnn = nn.LSTM(dec.prenet_dim + dec.encoder_embedding_dim,
                                     dec.attention_rnn_dim, 1)
        lstmcell2lstm_params(self.attention_rnn, dec.attention_rnn)
        self.attention_rnn.flatten_parameters()

        self.attention_layer = dec.attention_layer

        self.decoder_rnn = nn.LSTM(dec.attention_rnn_dim + dec.encoder_embedding_dim,
                                   dec.decoder_rnn_dim, 1)
        lstmcell2lstm_params(self.decoder_rnn, dec.decoder_rnn)
        self.decoder_rnn.flatten_parameters()

        self.linear_projection = dec.linear_projection
        self.gate_layer = dec.gate_layer


    def decode(self, decoder_input, in_attention_hidden, in_attention_cell,
               in_decoder_hidden, in_decoder_cell, in_attention_weights,
               in_attention_weights_cum, in_attention_context, memory,
               processed_memory, mask):

        cell_input = torch.cat((decoder_input, in_attention_context), -1)

        _, (out_attention_hidden, out_attention_cell) = self.attention_rnn(
            cell_input.unsqueeze(0), (in_attention_hidden.unsqueeze(0),
                                      in_attention_cell.unsqueeze(0)))
        out_attention_hidden = out_attention_hidden.squeeze(0)
        out_attention_cell = out_attention_cell.squeeze(0)

        out_attention_hidden = F.dropout(
            out_attention_hidden, self.p_attention_dropout, False)

        attention_weights_cat = torch.cat(
            (in_attention_weights.unsqueeze(1),
             in_attention_weights_cum.unsqueeze(1)), dim=1)
        out_attention_context, out_attention_weights = self.attention_layer(
            out_attention_hidden, memory, processed_memory,
            attention_weights_cat, mask)

        out_attention_weights_cum = in_attention_weights_cum + out_attention_weights
        decoder_input_tmp = torch.cat(
            (out_attention_hidden, out_attention_context), -1)

        _, (out_decoder_hidden, out_decoder_cell) = self.decoder_rnn(
            decoder_input_tmp.unsqueeze(0), (in_decoder_hidden.unsqueeze(0),
                                             in_decoder_cell.unsqueeze(0)))
        out_decoder_hidden = out_decoder_hidden.squeeze(0)
        out_decoder_cell = out_decoder_cell.squeeze(0)

        out_decoder_hidden = F.dropout(
            out_decoder_hidden, self.p_decoder_dropout, False)

        decoder_hidden_attention_context = torch.cat(
            (out_decoder_hidden, out_attention_context), 1)

        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return (decoder_output, gate_prediction, out_attention_hidden,
                out_attention_cell, out_decoder_hidden, out_decoder_cell,
                out_attention_weights, out_attention_weights_cum, out_attention_context)

    # @torch.jit.script
    def forward(self,
                decoder_input,
                attention_hidden,
                attention_cell,
                decoder_hidden,
                decoder_cell,
                attention_weights,
                attention_weights_cum,
                attention_context,
                memory,
                processed_memory,
                mask):
        decoder_input1 = self.prenet.infer(self.prenet, decoder_input)
        outputs = self.decode(decoder_input1,
                              attention_hidden,
                              attention_cell,
                              decoder_hidden,
                              decoder_cell,
                              attention_weights,
                              attention_weights_cum,
                              attention_context,
                              memory,
                              processed_memory,
                              mask)
        return outputs


def test_inference(encoder, decoder_iter, postnet):

    encoder.eval()
    decoder_iter.eval()
    postnet.eval()

    sys.path.append('./tensorrt')
    from inference_trt import init_decoder_inputs

    texts = ["Hello World, good day."]
    sequences, sequence_lengths = prepare_input_sequence(texts)

    measurements = {}

    print("Running Tacotron2 Encoder")
    with torch.no_grad():
        memory, processed_memory, lens = encoder(sequences, sequence_lengths)

    print("Running Tacotron2 Decoder")
    device = memory.device
    dtype = memory.dtype
    mel_lengths = torch.zeros([memory.size(0)], dtype=torch.int32, device = device)
    not_finished = torch.ones([memory.size(0)], dtype=torch.int32, device = device)
    mel_outputs, gate_outputs, alignments = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
    gate_threshold = 0.6
    max_decoder_steps = 1000
    first_iter = True

    (decoder_input, attention_hidden, attention_cell, decoder_hidden,
     decoder_cell, attention_weights, attention_weights_cum,
     attention_context, memory, processed_memory,
     mask) = init_decoder_inputs(memory, processed_memory, sequence_lengths)

    while True:
        with torch.no_grad():
            (mel_output, gate_output,
             attention_hidden, attention_cell,
             decoder_hidden, decoder_cell,
             attention_weights, attention_weights_cum,
             attention_context) = decoder_iter(decoder_input, attention_hidden, attention_cell, decoder_hidden,
                                               decoder_cell, attention_weights, attention_weights_cum,
                                               attention_context, memory, processed_memory, mask)

        if first_iter:
            mel_outputs = torch.unsqueeze(mel_output, 2)
            gate_outputs = torch.unsqueeze(gate_output, 2)
            alignments = torch.unsqueeze(attention_weights, 2)
            first_iter = False
        else:
            mel_outputs = torch.cat((mel_outputs, torch.unsqueeze(mel_output, 2)), 2)
            gate_outputs = torch.cat((gate_outputs, torch.unsqueeze(gate_output, 2)), 2)
            alignments = torch.cat((alignments, torch.unsqueeze(attention_weights, 2)), 2)

        dec = torch.le(torch.sigmoid(gate_output), gate_threshold).to(torch.int32).squeeze(1)
        not_finished = not_finished*dec
        mel_lengths += not_finished

        if torch.sum(not_finished) == 0:
            print("Stopping after ",mel_outputs.size(2)," decoder steps")
            break
        if mel_outputs.size(2) == max_decoder_steps:
            print("Warning! Reached max decoder steps")
            break

        decoder_input = mel_output


    print("Running Tacotron2 PostNet")
    with torch.no_grad():
        mel_outputs_postnet = postnet(mel_outputs)

    return mel_outputs_postnet

def main():

    parser = argparse.ArgumentParser(
        description='PyTorch Tacotron 2 export to TRT')
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()

    tacotron2 = load_and_setup_model('Tacotron2', parser, args.tacotron2,
                                     fp16_run=args.fp16, cpu_run=False)

    opset_version = 10

    sequences = torch.randint(low=0, high=148, size=(1,50),
                             dtype=torch.long).cuda()
    sequence_lengths = torch.IntTensor([sequences.size(1)]).cuda().long()
    dummy_input = (sequences, sequence_lengths)

    encoder = Encoder(tacotron2)
    encoder.eval()
    with torch.no_grad():
        encoder(*dummy_input)

    torch.onnx.export(encoder, dummy_input, args.output+"/"+"encoder.onnx",
                      opset_version=opset_version,
                      do_constant_folding=True,
                      input_names=["sequences", "sequence_lengths"],
                      output_names=["memory", "processed_memory", "lens"],
                      dynamic_axes={"sequences": {1: "text_seq"},
                                    "memory": {1: "mem_seq"},
                                    "processed_memory": {1: "mem_seq"}
                      })

    decoder_iter = DecoderIter(tacotron2)
    memory = torch.randn((1,sequence_lengths[0],512)).cuda() #encoder_outputs
    if args.fp16:
        memory = memory.half()
    memory_lengths = sequence_lengths
    # initialize decoder states for dummy_input
    decoder_input = tacotron2.decoder.get_go_frame(memory)
    mask = get_mask_from_lengths(memory_lengths)
    (attention_hidden,
     attention_cell,
     decoder_hidden,
     decoder_cell,
     attention_weights,
     attention_weights_cum,
     attention_context,
     processed_memory) = tacotron2.decoder.initialize_decoder_states(memory)
    dummy_input = (decoder_input,
                   attention_hidden,
                   attention_cell,
                   decoder_hidden,
                   decoder_cell,
                   attention_weights,
                   attention_weights_cum,
                   attention_context,
                   memory,
                   processed_memory,
                   mask)

    decoder_iter = DecoderIter(tacotron2)
    decoder_iter.eval()
    with torch.no_grad():
        decoder_iter(*dummy_input)

    torch.onnx.export(decoder_iter, dummy_input, args.output+"/"+"decoder_iter.onnx",
                      opset_version=opset_version,
                      do_constant_folding=True,
                      input_names=["decoder_input",
                                   "attention_hidden",
                                   "attention_cell",
                                   "decoder_hidden",
                                   "decoder_cell",
                                   "attention_weights",
                                   "attention_weights_cum",
                                   "attention_context",
                                   "memory",
                                   "processed_memory",
                                   "mask"],
                      output_names=["decoder_output",
                                    "gate_prediction",
                                    "out_attention_hidden",
                                    "out_attention_cell",
                                    "out_decoder_hidden",
                                    "out_decoder_cell",
                                    "out_attention_weights",
                                    "out_attention_weights_cum",
                                    "out_attention_context"],
                      dynamic_axes={"attention_weights" : {1: "seq_len"},
                                    "attention_weights_cum" : {1: "seq_len"},
                                    "memory" : {1: "seq_len"},
                                    "processed_memory" : {1: "seq_len"},
                                    "mask" : {1: "seq_len"},
                                    "out_attention_weights" : {1: "seq_len"},
                                    "out_attention_weights_cum" : {1: "seq_len"}
                      })

    postnet = Postnet(tacotron2)
    dummy_input = torch.randn((1,80,620)).cuda()
    if args.fp16:
        dummy_input = dummy_input.half()
    torch.onnx.export(postnet, dummy_input, args.output+"/"+"postnet.onnx",
                      opset_version=opset_version,
                      do_constant_folding=True,
                      input_names=["mel_outputs"],
                      output_names=["mel_outputs_postnet"],
                      dynamic_axes={"mel_outputs": {2: "mel_seq"},
                                    "mel_outputs_postnet": {2: "mel_seq"}})

    mel = test_inference(encoder, decoder_iter, postnet)
    torch.save(mel, "mel.pt")

if __name__ == '__main__':
    main()
