import logging
import os

import tensorflow as tf
import dllogger

from mrcnn_tf2.model.mask_rcnn import MaskRCNN
from mrcnn_tf2.runtime.callbacks import DLLoggerMetricsCallback, DLLoggerPerfCallback, PretrainedWeightsLoadingCallback
from mrcnn_tf2.runtime.evaluation import evaluate
from mrcnn_tf2.runtime.learning_rate import PiecewiseConstantWithWarmupSchedule
from mrcnn_tf2.runtime.weights_mapping import WEIGHTS_MAPPING


def run_training(dataset, params):
    setup(params)

    strategy = tf.distribute.MirroredStrategy()
    params.replicas = strategy.num_replicas_in_sync
    params.global_train_batch_size = params.train_batch_size * params.replicas
    logging.info(f'Distributed Strategy is activated for {params.replicas} device(s)')

    with strategy.scope():

        learning_rate = PiecewiseConstantWithWarmupSchedule(
            init_value=params.init_learning_rate,
            # scale boundaries from epochs to steps
            boundaries=[
                int(b * dataset.train_size / params.global_train_batch_size)
                for b in params.learning_rate_boundaries
            ],
            values=params.learning_rate_values,
            # scale only by local BS as distributed strategy later scales it by number of replicas
            scale=params.train_batch_size
        )

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=params.momentum
        )

        mask_rcnn_model = create_model(params)

        mask_rcnn_model.compile(
            optimizer=optimizer
        )

    # distributed strategy splits data between instances so we need global BS
    train_data = dataset.train_fn(batch_size=params.global_train_batch_size)

    if params.eagerly:
        mask_rcnn_model.run_eagerly = True
        logging.warning('Model is running in eager mode which might reduce performance')

    mask_rcnn_model.fit(
        x=train_data,
        epochs=params.epochs,
        steps_per_epoch=params.steps_per_epoch or (dataset.train_size // params.global_train_batch_size),
        callbacks=list(create_callbacks(params)),
        verbose=0
    )


def run_evaluation(dataset, params):
    setup(params)

    mask_rcnn_model = create_model(params)

    if params.eagerly:
        mask_rcnn_model.run_eagerly = True
        logging.warning('Model is running in eager mode which might reduce performance')

    predictions = mask_rcnn_model.predict(
        x=dataset.eval_fn(params.eval_batch_size),
        callbacks=list(create_callbacks(params))
    )

    eval_results = evaluate(
        predictions=predictions,
        eval_file=params.eval_file,
        include_mask=params.include_mask
    )

    dllogger.log(
        step=tuple(),
        data={k: float(v) for k, v in eval_results.items()}
    )


def run_inference(dataset, params):
    setup(params)

    mask_rcnn_model = create_model(params)

    if params.eagerly:
        mask_rcnn_model.run_eagerly = True
        logging.warning('Model is running in eager mode which might reduce performance')

    mask_rcnn_model.predict(
        x=dataset.eval_fn(params.eval_batch_size),
        callbacks=list(create_callbacks(params))
    )


def setup(params):

    # enforces that AMP is enabled using --amp and not env var
    # mainly for NGC where it is enabled by default
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

    if params.xla:
        tf.config.optimizer.set_jit(True)
        logging.info('XLA is activated')

    if params.amp:
        policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16", loss_scale="dynamic")
        tf.keras.mixed_precision.experimental.set_policy(policy)
        logging.info('AMP is activated')


def create_model(params):
    model = MaskRCNN(
        params=params,
        trainable='train' in params.mode
    )

    checkpoint_path = tf.train.latest_checkpoint(params.model_dir)

    # if there is no checkpoint we are done
    if checkpoint_path is None:
        logging.info(f"No checkpoint was found in: {params.model_dir}")
        return model

    model.load_weights(checkpoint_path).expect_partial()
    logging.info(f"Loaded weights from checkpoint: {checkpoint_path}")

    # don't load backbone weights to do not override the checkpoint
    if params.backbone_checkpoint:
        params.backbone_checkpoint = None
        logging.info("Pretrained backbone weights will not be loaded")

    return model


def create_callbacks(params):
    yield DLLoggerMetricsCallback(
        dllogger=dllogger,
        log_every=params.log_every
    )

    yield DLLoggerPerfCallback(
        dllogger=dllogger,
        batch_sizes={
            'train': params.train_batch_size * getattr(params, 'replicas', 1),
            'test': params.eval_batch_size * getattr(params, 'replicas', 1),
            'predict': params.eval_batch_size * getattr(params, 'replicas', 1)
        },
        warmup_steps=params.log_warmup_steps,
        log_every=params.log_every
    )

    if params.backbone_checkpoint:
        yield PretrainedWeightsLoadingCallback(
            checkpoint_path=params.backbone_checkpoint,
            mapping=lambda name: WEIGHTS_MAPPING.get(name.replace(':0', ''), name)
        )

    yield tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(params.model_dir, params.checkpoint_name_format),
        verbose=1
    )

    if params.log_tensorboard:
        yield tf.keras.callbacks.TensorBoard(
            log_dir=params.log_tensorboard,
            update_freq='batch'
        )
