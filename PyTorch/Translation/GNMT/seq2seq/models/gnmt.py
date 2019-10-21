# Copyright (c) 2017 Elad Hoffer
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch.nn as nn

import seq2seq.data.config as config
from seq2seq.models.decoder import ResidualRecurrentDecoder
from seq2seq.models.encoder import ResidualRecurrentEncoder
from seq2seq.models.seq2seq_base import Seq2Seq


class GNMT(Seq2Seq):
    """
    GNMT v2 model
    """
    def __init__(self, vocab_size, hidden_size=1024, num_layers=4, dropout=0.2,
                 batch_first=False, share_embedding=True):
        """
        Constructor for the GNMT v2 model.

        :param vocab_size: size of vocabulary (number of tokens)
        :param hidden_size: internal hidden size of the model
        :param num_layers: number of layers, applies to both encoder and
            decoder
        :param dropout: probability of dropout (in encoder and decoder)
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param share_embedding: if True embeddings are shared between encoder
            and decoder
        """

        super(GNMT, self).__init__(batch_first=batch_first)

        if share_embedding:
            embedder = nn.Embedding(vocab_size, hidden_size,
                                    padding_idx=config.PAD)
            nn.init.uniform_(embedder.weight.data, -0.1, 0.1)
        else:
            embedder = None

        self.encoder = ResidualRecurrentEncoder(vocab_size, hidden_size,
                                                num_layers, dropout,
                                                batch_first, embedder)

        self.decoder = ResidualRecurrentDecoder(vocab_size, hidden_size,
                                                num_layers, dropout,
                                                batch_first, embedder)

    def forward(self, input_encoder, input_enc_len, input_decoder):
        context = self.encode(input_encoder, input_enc_len)
        context = (context, input_enc_len, None)
        output, _, _ = self.decode(input_decoder, context)

        return output
