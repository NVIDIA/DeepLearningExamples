# Copyright (c) 2018. All rights reserved.
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
#
# -----------------------------------------------------------------------
#
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

import tensorflow as tf
import horovod.tensorflow as hvd

def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """
    Custom variable getter that forces trainable variables to be stored in
    float32 precision and then casts them to the half-precision
    """
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable

def neural_mf(users,
              items,
              model_dtype,
              nb_users,
              nb_items,
              mf_dim,
              mf_reg,
              mlp_layer_sizes,
              mlp_layer_regs,
              dropout_rate,
              sigmoid=False):
    """
    Constructs the model graph
    """
    # Check params
    if len(mlp_layer_sizes) != len(mlp_layer_regs):
        raise RuntimeError('u dummy, layer_sized != layer_regs')
    if mlp_layer_sizes[0] % 2 != 0:
        raise RuntimeError('u dummy, mlp_layer_sizes[0] % 2 != 0')
    nb_mlp_layers = len(mlp_layer_sizes)

    # Embeddings
    user_embed = tf.get_variable(
        "user_embeddings",
        shape=[nb_users, mf_dim + mlp_layer_sizes[0] // 2],
        initializer=tf.initializers.random_normal(mean=0.0, stddev=0.01))
    item_embed = tf.get_variable(
        "item_embeddings",
        shape=[nb_items, mf_dim + mlp_layer_sizes[0] // 2],
        initializer=tf.initializers.random_normal(mean=0.0, stddev=0.01))
    # Matrix Factorization Embeddings
    xmfu = tf.nn.embedding_lookup(user_embed[:, :mf_dim], users, partition_strategy='div')
    xmfi = tf.nn.embedding_lookup(item_embed[:, :mf_dim], items, partition_strategy='div')
    # MLP Network Embeddings
    xmlpu = tf.nn.embedding_lookup(user_embed[:, mf_dim:], users, partition_strategy='div')
    xmlpi = tf.nn.embedding_lookup(item_embed[:, mf_dim:], items, partition_strategy='div')
    # Enforce model to use fp16 data types when manually enabling mixed precision
    # (Tensorfow ops will use automatically use the data type of the first input)
    if model_dtype == tf.float16:
        xmfu = tf.cast(xmfu, model_dtype)
        xmfi = tf.cast(xmfi, model_dtype)
        xmlpu = tf.cast(xmlpu, model_dtype)
        xmlpi = tf.cast(xmlpi, model_dtype)

    # Matrix Factorization
    xmf = tf.math.multiply(xmfu, xmfi)

    # MLP Layers
    xmlp = tf.concat((xmlpu, xmlpi), 1)
    for i in range(1, nb_mlp_layers):
        xmlp = tf.layers.Dense(
            mlp_layer_sizes[i],
            activation=tf.nn.relu,
            kernel_initializer=tf.glorot_uniform_initializer()
        ).apply(xmlp)
        xmlp = tf.layers.Dropout(rate=dropout_rate).apply(xmlp)

    # Final fully-connected layer
    logits = tf.concat((xmf, xmlp), 1)
    logits = tf.layers.Dense(
        1,
        kernel_initializer=tf.keras.initializers.lecun_uniform()
    ).apply(logits)

    if sigmoid:
        logits = tf.math.sigmoid(logits)

    # Cast model outputs back to float32 if manually enabling mixed precision for loss calculation
    if model_dtype == tf.float16:
        logits = tf.cast(logits, tf.float32)

    return logits

def compute_eval_metrics(logits, dup_mask, val_batch_size, K):
    """
    Constructs the graph to compute Hit Rate and NDCG
    """
    # Replace duplicate (uid, iid) pairs with -inf
    logits = logits * (1. - dup_mask)
    logits = logits + (dup_mask * logits.dtype.min)
    # Reshape tensors so that each row corresponds with a user
    logits_by_user = tf.reshape(logits, [-1, val_batch_size])
    dup_mask_by_user = tf.cast(tf.reshape(logits, [-1, val_batch_size]), tf.bool)
    # Get the topk items for each user
    top_item_indices = tf.math.top_k(logits_by_user, K)[1]
    # Check that the positive sample (last index) is in the top K
    is_positive = tf.cast(tf.equal(top_item_indices, val_batch_size-1), tf.int32)
    found_positive = tf.reduce_sum(is_positive, axis=1)
    # Extract the rankings of the positive samples
    positive_ranks = tf.reduce_sum(is_positive * tf.expand_dims(tf.range(K), 0), axis=1)
    dcg = tf.log(2.) / tf.log(tf.cast(positive_ranks, tf.float32) + 2)
    dcg *= tf.cast(found_positive, dcg.dtype)

    return found_positive, dcg

def ncf_model_ops(users,
                  items,
                  labels,
                  dup_mask,
                  params,
                  mode='TRAIN'):
    """
    Constructs the training and evaluation graphs
    """
    # Validation params
    val_batch_size = params['val_batch_size']
    K = params['top_k']
    # Training params
    learning_rate = params['learning_rate']
    beta_1 = params['beta_1']
    beta_2 = params['beta_2']
    epsilon = params['epsilon']
    # Model params
    fp16 = params['fp16']
    nb_users = params['num_users']
    nb_items = params['num_items']
    mf_dim = params['num_factors']
    mf_reg = params['mf_reg']
    mlp_layer_sizes = params['layer_sizes']
    mlp_layer_regs = params['layer_regs']
    dropout = params['dropout']
    sigmoid = False #params['sigmoid']
    loss_scale = params['loss_scale']

    model_dtype = tf.float16 if fp16 else tf.float32

    # If manually enabling mixed precision, use the custom variable getter
    custom_getter = None if not fp16 else float32_variable_storage_getter
    # Allow soft device placement
    with tf.device(None), \
         tf.variable_scope('neumf', custom_getter=custom_getter):
        # Model graph
        logits = neural_mf(
            users,
            items,
            model_dtype,
            nb_users,
            nb_items,
            mf_dim,
            mf_reg,
            mlp_layer_sizes,
            mlp_layer_regs,
            dropout,
            sigmoid
        )
        logits = tf.squeeze(logits)

        if mode == 'INFERENCE':
            return logits

        # Evaluation Ops
        found_positive, dcg = compute_eval_metrics(logits, dup_mask, val_batch_size, K)
        # Metrics
        hit_rate = tf.metrics.mean(found_positive, name='hit_rate')
        ndcg = tf.metrics.mean(dcg, name='ndcg')

        eval_op = tf.group(hit_rate[1], ndcg[1])

        if mode == 'EVAL':
            return hit_rate[0], ndcg[0], eval_op, None

        # Labels
        labels = tf.reshape(labels, [-1, 1])
        logits = tf.reshape(logits, [-1, 1])

        # Use adaptive momentum optimizer
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=beta_1, beta2=beta_2,
            epsilon=epsilon)

        loss = tf.losses.sigmoid_cross_entropy(
            labels,
            logits,
            reduction=tf.losses.Reduction.MEAN)

        # Apply loss scaling if manually enabling mixed precision
        if fp16:
            if loss_scale is None:
                loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(2**32, 1000)
            else:
                loss_scale_manager = tf.contrib.mixed_precision.FixedLossScaleManager(loss_scale)
            optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(optimizer, loss_scale_manager)

        # Horovod wrapper for distributed training
        optimizer = hvd.DistributedOptimizer(optimizer)

        # Update ops
        global_step = tf.train.get_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

        return hit_rate[0], ndcg[0], eval_op, train_op
