# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from torch.nn import MSELoss, KLDivLoss, CosineEmbeddingLoss
import math


class TransformerLosses():
    """Implements transformer specific loss functions for Knowledge Distillation.
    """

    def __init__(self, student_config, teacher_config, device, args):
        self.mse_loss = MSELoss()
        self.kl_loss = KLDivLoss(reduction='batchmean')
        self.cosine_loss = CosineEmbeddingLoss()
        self.distill_config = student_config.distillation_config
        self.device = device
        self.student_config = student_config
        self.teacher_config = teacher_config
        self.batch_size = args.train_batch_size

    def compute_loss_(self, pred, target, loss_name):

        if self.distill_config[loss_name] == "mse":
            return self.mse_loss(pred, target)

        elif self.distill_config[loss_name] == "kld":
            seq_length = pred.size(0) if loss_name == "value_state_loss" else pred.size(-1)
            if loss_name == "value_state_loss":
                dk_student = pred.shape[-1] // self.student_config.num_attention_heads
                dk_teacher = target.shape[-1] // self.teacher_config.num_attention_heads
                # Old: (bsz, seq, heads * dk) => (bsz, heads, seq, dk)
                # New: (seq, bsz, heads * dk) => (bsz * heads, seq, dk)
                student_values = pred.view(seq_length, self.batch_size * self.student_config.num_attention_heads,
                                           dk_student)
                student_values = student_values.transpose(0, 1)
                teacher_values = target.view(seq_length, self.batch_size * self.teacher_config.num_attention_heads,
                                             dk_teacher)
                teacher_values = teacher_values.transpose(0, 1)
                # (..., seq, dk) x (..., dk, seq)
                pred = torch.bmm(student_values, student_values.transpose(1, 2)) / math.sqrt(dk_student)
                target = torch.bmm(teacher_values, teacher_values.transpose(1, 2)) / math.sqrt(dk_teacher)

                pred = pred.view(self.batch_size, self.student_config.num_attention_heads, seq_length, seq_length)
                target = target.view(self.batch_size, self.teacher_config.num_attention_heads, seq_length, seq_length)

            return self.kl_loss(torch.nn.LogSoftmax(dim=-1)(pred), torch.nn.Softmax(dim=-1)(target)) / (
                        self.student_config.num_attention_heads * seq_length)

        elif self.distill_config[loss_name] == "cosine":
            # seq_length = pred.size(0)
            # return self.cosine_loss(pred.transpose(0, 2).reshape(-1, seq_length),
            #                         target.transpose(0, 2).reshape(-1, seq_length),
            #                         torch.tensor([1]).to(self.device))
            return self.cosine_loss(pred.view(-1, self.teacher_config.hidden_size),
                                    target.view(-1, self.teacher_config.hidden_size),
                                    torch.tensor([1]).to(self.device))

        else:
            error_string = "'attention_loss':{} not defined. Choose among 'mse', 'cosine' or 'kld'".format(
                self.distill_config["attention_loss"])
            raise ValueError(error_string)

    def compute_loss(self, pred, target, loss_name):

        loss = 0.
        for student, teacher in zip(pred, target):
            if loss_name == "attention_loss":
                student = torch.where(student <= -1e2, torch.zeros_like(student).to(self.device),
                                      student)
                teacher = torch.where(teacher <= -1e2, torch.zeros_like(teacher).to(self.device),
                                      teacher)
            tmp_loss = self.compute_loss_(student, teacher, loss_name)
            loss += tmp_loss
        return loss
