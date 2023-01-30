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

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
from torch.nn import (
    BatchNorm2d,
    BCELoss,
    Conv2d,
    ConvTranspose2d,
    CrossEntropyLoss,
    Dropout,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
    SmoothL1Loss,
)
from torch.nn import functional as F
from torch.nn import init
from torch.optim import Adam
from sklearn import model_selection, preprocessing

from syngen.generator.tabular.base_tabular_generator import BaseTabularGenerator
from syngen.generator.tabular.data_transformer.ctab_data_transformer import (
    CTABDataTransformer,
    ImageTransformer,
)
from syngen.utils.types import ColumnType


class CTABGenerator(BaseTabularGenerator):
    """
    Adopted from: https://github.com/Team-TUD/CTAB-GAN
    Args:

        embedding_dim (int): Size of the random sample passed to the Generator. Defaults to 128.
        classifier_dim (tuple or list of ints): Size of the output samples for each one of the classifier Layers.
        A Linear Layer will be created for each one of the values provided.
        Defaults to (256, 256).
        l2scale (float): L2 regularization scaling. Defaults to 1e-5.
        batch_size (int): Number of data samples to process in each step.
        epochs (int): Number of training epochs. Defaults to 300.
    """

    def __init__(
        self,
        classifier_dim: Tuple[int] = (256, 256, 256, 256),
        embedding_dim: int = 100,
        num_channels: int = 64,
        l2scale: float = 1e-5,
        batch_size: int = 500,
        epochs: int = 1,
        test_ratio: float = 0.1,
        **kwargs,
    ):

        self.embedding_dim = embedding_dim
        self.classifier_dim = classifier_dim
        self.num_channels = num_channels
        self.dside = None
        self.gside = None
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self._device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.test_ratio = test_ratio

    def column_check(self, data: pd.DataFrame, columns: list):
        data_cols = data.columns
        invalid_cols = []
        for c in columns:
            if c not in data_cols:
                invalid_cols.append(c)
        return invalid_cols

    def set_device(self, device):
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)

    def fit(
        self,
        train_data: pd.DataFrame,
        categorical_columns: List[str] = [],
        log_columns: List[str] = [],
        integer_columns: List[str] = [],
        mixed_columns: Dict = {},
        problem_type: Dict = {},
    ):

        specified_cols = (
            list(categorical_columns)
            + list(log_columns)
            + list(mixed_columns)
            + list(integer_columns)
        )
        target_col = None
        target_index = None
        if problem_type:  # - supports only single problem type
            target_col = list(problem_type.values())[0]
            specified_cols += [target_col]

        # - check for invalid columns
        invalid_cols = self.column_check(train_data, specified_cols)
        if len(invalid_cols):
            raise ValueError(f"invalid columns: {invalid_cols}")

        if target_col is not None:
            target_index = train_data.columns.get_loc(target_col)

        self.data_prep = DataPreprocessing(
            categorical_columns=categorical_columns,
            log_columns=log_columns,
            mixed_columns=mixed_columns,
            integer_columns=integer_columns,
            test_ratio=self.test_ratio,
            target_col=target_col,
        )
        train_data = self.data_prep.transform(train_data)
        categorical_columns = self.data_prep.column_types[
            ColumnType.CATEGORICAL
        ]
        mixed_columns = self.data_prep.column_types[ColumnType.MIXED]
        self.transformer = CTABDataTransformer(
            categorical_columns=categorical_columns, mixed_dict=mixed_columns
        )
        self.transformer.fit(train_data)

        train_data = self.transformer.transform(train_data.values)

        data_sampler = Sampler(train_data, self.transformer.output_info)
        data_dim = self.transformer.output_dim
        self.cond_generator = Cond(train_data, self.transformer.output_info)
        sides = [4, 8, 16, 24, 32, 64, 128]
        col_size_d = data_dim + self.cond_generator.n_opt
        for i in sides:
            if i * i >= col_size_d:
                self.dside = i
                break

        sides = [4, 8, 16, 24, 32, 64, 128]
        col_size_g = data_dim
        for i in sides:
            if i * i >= col_size_g:
                self.gside = i
                break

        layers_G = determine_layers_gen(
            self.gside,
            self.embedding_dim + self.cond_generator.n_opt,
            self.num_channels,
        )
        layers_D = determine_layers_disc(self.dside, self.num_channels)

        self._generator = Generator(self.gside, layers_G).to(self._device)
        discriminator = Discriminator(self.dside, layers_D).to(self._device)
        optimizer_params = dict(
            lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale
        )
        optimizerG = Adam(self._generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)

        st_ed = None
        classifier = None
        optimizerC = None
        if target_index is not None:
            st_ed = get_st_ed(target_index, self.transformer.output_info)
            classifier = Classifier(data_dim, self.classifier_dim, st_ed).to(
                self._device
            )
            optimizerC = optim.Adam(
                classifier.parameters(), **optimizer_params
            )

        self._generator.apply(weights_init)
        discriminator.apply(weights_init)

        self.Gtransformer = ImageTransformer(self.gside)
        self.Dtransformer = ImageTransformer(self.dside)

        steps_per_epoch = max(1, len(train_data) // self.batch_size)
        for i in range(self.epochs):
            for _ in range(steps_per_epoch):

                noisez = torch.randn(
                    self.batch_size, self.embedding_dim, device=self._device
                )
                condvec = self.cond_generator.sample_train(self.batch_size)

                c, m, col, opt = condvec
                c = torch.from_numpy(c).to(self._device)
                m = torch.from_numpy(m).to(self._device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez = noisez.view(
                    self.batch_size,
                    self.embedding_dim + self.cond_generator.n_opt,
                    1,
                    1,
                )

                perm = np.arange(self.batch_size)
                np.random.shuffle(perm)
                real = data_sampler.sample(
                    self.batch_size, col[perm], opt[perm]
                )
                c_perm = c[perm]

                real = torch.from_numpy(real.astype("float32")).to(
                    self._device
                )
                fake = self._generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, self.transformer.output_info)

                fake_cat = torch.cat([fakeact, c], dim=1)
                real_cat = torch.cat([real, c_perm], dim=1)

                real_cat_d = self.Dtransformer.transform(real_cat)
                fake_cat_d = self.Dtransformer.transform(fake_cat)

                optimizerD.zero_grad()
                y_real, _ = discriminator(real_cat_d)
                y_fake, _ = discriminator(fake_cat_d)
                loss_d = -(torch.log(y_real + 1e-4).mean()) - (
                    torch.log(1.0 - y_fake + 1e-4).mean()
                )
                loss_d.backward()
                optimizerD.step()

                noisez = torch.randn(
                    self.batch_size, self.embedding_dim, device=self._device
                )

                condvec = self.cond_generator.sample_train(self.batch_size)

                c, m, col, opt = condvec
                c = torch.from_numpy(c).to(self._device)
                m = torch.from_numpy(m).to(self._device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez = noisez.view(
                    self.batch_size,
                    self.embedding_dim + self.cond_generator.n_opt,
                    1,
                    1,
                )

                optimizerG.zero_grad()

                fake = self._generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, self.transformer.output_info)

                fake_cat = torch.cat([fakeact, c], dim=1)
                fake_cat = self.Dtransformer.transform(fake_cat)

                y_fake, info_fake = discriminator(fake_cat)

                cross_entropy = cond_loss(
                    faket, self.transformer.output_info, c, m
                )

                _, info_real = discriminator(real_cat_d)

                g = -(torch.log(y_fake + 1e-4).mean()) + cross_entropy
                g.backward(retain_graph=True)
                loss_mean = torch.norm(
                    torch.mean(info_fake.view(self.batch_size, -1), dim=0)
                    - torch.mean(info_real.view(self.batch_size, -1), dim=0),
                    1,
                )
                loss_std = torch.norm(
                    torch.std(info_fake.view(self.batch_size, -1), dim=0)
                    - torch.std(info_real.view(self.batch_size, -1), dim=0),
                    1,
                )
                loss_info = loss_mean + loss_std
                loss_info.backward()
                optimizerG.step()

                if problem_type:
                    fake = self._generator(noisez)
                    faket = self.Gtransformer.inverse_transform(fake)
                    fakeact = apply_activate(
                        faket, self.transformer.output_info
                    )

                    real_pre, real_label = classifier(real)
                    fake_pre, fake_label = classifier(fakeact)

                    c_loss = CrossEntropyLoss()

                    if (st_ed[1] - st_ed[0]) == 1:
                        c_loss = SmoothL1Loss()
                        real_label = real_label.type_as(real_pre)
                        fake_label = fake_label.type_as(fake_pre)
                        real_label = torch.reshape(real_label, real_pre.size())
                        fake_label = torch.reshape(fake_label, fake_pre.size())

                    elif (st_ed[1] - st_ed[0]) == 2:
                        c_loss = BCELoss()
                        real_label = real_label.type_as(real_pre)
                        fake_label = fake_label.type_as(fake_pre)

                    loss_cc = c_loss(real_pre, real_label)
                    loss_cg = c_loss(fake_pre, fake_label)

                    optimizerG.zero_grad()
                    loss_cg.backward()
                    optimizerG.step()

                    optimizerC.zero_grad()
                    loss_cc.backward()
                    optimizerC.step()

    def sample(self, n, **kwargs):
        assert hasattr(self, "_generator"), "`fit` function must be called prior to `sample`"

        self._generator.eval()

        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1

        data = []

        for i in range(steps):
            noisez = torch.randn(
                self.batch_size, self.embedding_dim, device=self._device
            )
            condvec = self.cond_generator.sample(self.batch_size)
            c = condvec
            c = torch.from_numpy(c).to(self._device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez = noisez.view(
                self.batch_size,
                self.embedding_dim + self.cond_generator.n_opt,
                1,
                1,
            )

            fake = self._generator(noisez)
            faket = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket, output_info)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        result = self.transformer.inverse_transform(data)
        output = self.data_prep.inverse_prep(result)
        return output.iloc[:n]


class Classifier(Module):
    def __init__(self, input_dim, dis_dims, st_ed):
        super(Classifier, self).__init__()
        dim = input_dim - (st_ed[1] - st_ed[0])
        seq = []
        self.str_end = st_ed
        for item in list(dis_dims):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        if (st_ed[1] - st_ed[0]) == 1:
            seq += [Linear(dim, 1)]

        elif (st_ed[1] - st_ed[0]) == 2:
            seq += [Linear(dim, 1), Sigmoid()]
        else:
            seq += [Linear(dim, (st_ed[1] - st_ed[0]))]

        self.seq = Sequential(*seq)

    def forward(self, input):

        label = None

        if (self.str_end[1] - self.str_end[0]) == 1:
            label = input[:, self.str_end[0] : self.str_end[1]]
        else:
            label = torch.argmax(
                input[:, self.str_end[0] : self.str_end[1]], axis=-1
            )

        new_imp = torch.cat(
            (input[:, : self.str_end[0]], input[:, self.str_end[1] :]), 1
        )

        if ((self.str_end[1] - self.str_end[0]) == 2) | (
            (self.str_end[1] - self.str_end[0]) == 1
        ):
            return self.seq(new_imp).view(-1), label
        else:
            return self.seq(new_imp), label


def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == "tanh":
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == "softmax":
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
    return torch.cat(data_t, dim=1)


def get_st_ed(target_col_index, output_info):
    st = 0
    c = 0
    tc = 0
    for item in output_info:
        if c == target_col_index:
            break
        if item[1] == "tanh":
            st += item[0]
        elif item[1] == "softmax":
            st += item[0]
            c += 1
        tc += 1
    ed = st + output_info[tc][0]
    return (st, ed)


def random_choice_prob_index_sampling(probs, col_idx):
    option_list = []
    for i in col_idx:
        pp = probs[i]
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))

    return np.array(option_list).reshape(col_idx.shape)


def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)


def maximum_interval(output_info):
    max_interval = 0
    for item in output_info:
        max_interval = max(max_interval, item[0])
    return max_interval


class Cond(object):
    def __init__(self, data, output_info):

        self.model = []
        st = 0
        counter = 0
        for item in output_info:

            if item[1] == "tanh":
                st += item[0]
            elif item[1] == "softmax":
                ed = st + item[0]
                counter += 1
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                st = ed

        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        st = 0
        self.p = np.zeros((counter, maximum_interval(output_info)))
        self.p_sampling = []
        for item in output_info:
            if item[1] == "tanh":
                st += item[0]
            elif item[1] == "softmax":
                ed = st + item[0]
                tmp = np.sum(data[:, st:ed], axis=0)
                tmp_sampling = np.sum(data[:, st:ed], axis=0)
                tmp = np.log(tmp + 1)
                tmp = tmp / np.sum(tmp)
                tmp_sampling = tmp_sampling / np.sum(tmp_sampling)
                self.p_sampling.append(tmp_sampling)
                self.p[self.n_col, : item[0]] = tmp
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                st = ed

        self.interval = np.asarray(self.interval)

    def sample_train(self, batch):
        if self.n_col == 0:
            return None
        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype="float32")
        mask = np.zeros((batch, self.n_col), dtype="float32")
        mask[np.arange(batch), idx] = 1
        opt1prime = random_choice_prob_index(self.p[idx])
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1

        return vec, mask, idx, opt1prime

    def sample(self, batch):
        if self.n_col == 0:
            return None
        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype="float32")
        opt1prime = random_choice_prob_index_sampling(self.p_sampling, idx)

        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1

        return vec


def cond_loss(data, output_info, c, m):
    loss = []
    st = 0
    st_c = 0
    for item in output_info:
        if item[1] == "tanh":
            st += item[0]

        elif item[1] == "softmax":
            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(
                data[:, st:ed],
                torch.argmax(c[:, st_c:ed_c], dim=1),
                reduction="none",
            )
            loss.append(tmp)
            st = ed
            st_c = ed_c

    loss = torch.stack(loss, dim=1)
    return (loss * m).sum() / data.size()[0]


class Sampler(object):
    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
        self.data = data
        self.model = []
        self.n = len(data)
        st = 0
        for item in output_info:
            if item[1] == "tanh":
                st += item[0]
            elif item[1] == "softmax":
                ed = st + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st + j])[0])
                self.model.append(tmp)
                st = ed

    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        return self.data[idx]


class Discriminator(Module):
    def __init__(self, side, layers):
        super(Discriminator, self).__init__()
        self.side = side
        info = len(layers) - 2
        self.seq = Sequential(*layers)
        self.seq_info = Sequential(*layers[:info])

    def forward(self, input):
        return (self.seq(input)), self.seq_info(input)


class Generator(Module):
    def __init__(self, side, layers):
        super(Generator, self).__init__()
        self.side = side
        self.seq = Sequential(*layers)

    def forward(self, input_):
        return self.seq(input_)


def determine_layers_disc(side, num_channels):
    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True),
        ]
    print()
    layers_D += [
        Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0),
        Sigmoid(),
    ]

    return layers_D


def determine_layers_gen(side, embedding_dim, num_channels):

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layers_G = [
        ConvTranspose2d(
            embedding_dim,
            layer_dims[-1][0],
            layer_dims[-1][1],
            1,
            0,
            output_padding=0,
            bias=False,
        )
    ]

    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [
            BatchNorm2d(prev[0]),
            ReLU(True),
            ConvTranspose2d(
                prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True
            ),
        ]
    return layers_G


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)


class DataPreprocessing(object):
    def __init__(
        self,
        categorical_columns: list,
        log_columns: list,
        mixed_columns: dict,
        integer_columns: list,
        test_ratio: float,
        target_col: str = None,
    ):
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.column_types = dict()
        self.column_types[ColumnType.CATEGORICAL] = []
        self.column_types[ColumnType.MIXED] = {}
        self.lower_bounds = {}
        self.label_encoder_list = []
        self.CONSTANT_INT = -9999999

        if target_col is not None:
            self.target_col = target_col

        self.test_ratio = test_ratio
        super().__init__()

    def transform(self, raw_df: pd.DataFrame):

        if hasattr(self, "target_col"):
            y_real = raw_df[self.target_col]
            X_real = raw_df.drop(columns=[self.target_col])
            (
                X_train_real,
                _,
                y_train_real,
                _,
            ) = model_selection.train_test_split(
                X_real,
                y_real,
                test_size=self.test_ratio,
                stratify=y_real,
                random_state=42,
            )
            X_train_real.loc[:, self.target_col] = y_train_real
        else:
            X_train_real = raw_df
        self.df = X_train_real
        self.df = self.df.replace(r" ", np.nan)
        self.df = self.df.fillna("empty")

        all_columns = set(self.df.columns)
        irrelevant_missing_columns = set(self.categorical_columns)
        relevant_missing_columns = list(
            all_columns - irrelevant_missing_columns
        )

        for i in relevant_missing_columns:
            if i in self.log_columns:
                if "empty" in list(self.df[i].values):
                    self.df[i] = self.df[i].apply(
                        lambda x: self.CONSTANT_INT if x == "empty" else x
                    )
                    self.mixed_columns[i] = [self.CONSTANT_INT]
            elif i in list(self.mixed_columns.keys()):
                if "empty" in list(self.df[i].values):
                    self.df[i] = self.df[i].apply(
                        lambda x: self.CONSTANT_INT if x == "empty" else x
                    )
                    self.mixed_columns[i].append(self.CONSTANT_INT)
            else:
                if "empty" in list(self.df[i].values):
                    self.df[i] = self.df[i].apply(
                        lambda x: self.CONSTANT_INT if x == "empty" else x
                    )
                    self.mixed_columns[i] = [self.CONSTANT_INT]

        if self.log_columns:
            for log_column in self.log_columns:
                valid_indices = []
                for idx, val in enumerate(self.df[log_column].values):
                    if val != self.CONSTANT_INT:
                        valid_indices.append(idx)
                eps = 1
                lower = np.min(self.df[log_column].iloc[valid_indices].values)
                self.lower_bounds[log_column] = lower
                if lower > 0:
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x) if x != self.CONSTANT_INT else self.CONSTANT_INT
                    )
                elif lower == 0:
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x + eps)
                        if x != self.CONSTANT_INT
                        else self.CONSTANT_INT
                    )
                else:
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x - lower + eps)
                        if x != self.CONSTANT_INT
                        else self.CONSTANT_INT
                    )

        for column_index, column in enumerate(self.df.columns):
            if column in self.categorical_columns:
                label_encoder = preprocessing.LabelEncoder()
                self.df[column] = self.df[column].astype(str)
                label_encoder.fit(self.df[column])
                current_label_encoder = dict()
                current_label_encoder["column"] = column
                current_label_encoder["label_encoder"] = label_encoder
                transformed_column = label_encoder.transform(self.df[column])
                self.df[column] = transformed_column
                self.label_encoder_list.append(current_label_encoder)
                self.column_types[ColumnType.CATEGORICAL].append(column_index)

            elif column in self.mixed_columns:
                self.column_types[ColumnType.MIXED][
                    column_index
                ] = self.mixed_columns[column]

        return self.df

    def inverse_prep(self, data, eps=1):

        df_sample = pd.DataFrame(data, columns=self.df.columns)

        for i in range(len(self.label_encoder_list)):
            le = self.label_encoder_list[i]["label_encoder"]
            df_sample[self.label_encoder_list[i]["column"]] = df_sample[
                self.label_encoder_list[i]["column"]
            ].astype(int)
            df_sample[
                self.label_encoder_list[i]["column"]
            ] = le.inverse_transform(
                df_sample[self.label_encoder_list[i]["column"]]
            )

        if self.log_columns:
            for i in df_sample:
                if i in self.log_columns:
                    lower_bound = self.lower_bounds[i]
                    if lower_bound > 0:
                        df_sample[i].apply(lambda x: np.exp(x))
                    elif lower_bound == 0:
                        df_sample[i] = df_sample[i].apply(
                            lambda x: np.ceil(np.exp(x) - eps)
                            if (np.exp(x) - eps) < 0
                            else (np.exp(x) - eps)
                        )
                    else:
                        df_sample[i] = df_sample[i].apply(
                            lambda x: np.exp(x) - eps + lower_bound
                        )

        if self.integer_columns:
            for column in self.integer_columns:
                df_sample[column] = np.round(df_sample[column].values)
                df_sample[column] = df_sample[column].astype(int)

        df_sample.replace(self.CONSTANT_INT, np.nan, inplace=True)
        df_sample.replace("empty", np.nan, inplace=True)

        return df_sample