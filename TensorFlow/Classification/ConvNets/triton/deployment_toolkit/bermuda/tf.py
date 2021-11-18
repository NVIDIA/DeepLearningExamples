# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np

# pytype: disable=import-error
import tensorflow as tf
from tensorflow.python.eager import wrap_function
from tf2onnx.shape_inference import infer_shape
from tf2onnx.tf_loader import (
    freeze_session,
    from_function,
    inputs_without_resource,
    is_function,
    remove_redundant_inputs,
    tf_optimize,
)

# pytype: enable=import-error

from ..args import filter_fn_args
from ..core import (
    GET_MODEL_FN_NAME,
    GET_SERVING_INPUT_RECEIVER_FN,
    BaseConverter,
    BaseLoader,
    BaseRunner,
    BaseRunnerSession,
    BaseSaver,
    Format,
    Model,
    Precision,
    TensorSpec,
    load_from_file,
)
from ..extensions import converters, loaders, runners, savers
from .utils import infer_precision

LOGGER = logging.getLogger(__name__)


def is_tf2():
    return tf.__version__.startswith("2.")


def create_session_config(*, allow_growth=False, use_xla=False, gpu_memory_fraction=1.0):
    gpu_options = tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=gpu_memory_fraction, allow_growth=allow_growth
    )
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    if use_xla:
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    LOGGER.debug(
        f"Using gpu memory fraction: allow_growth={allow_growth} "
        f"gpu_memory_fraction={gpu_memory_fraction} "
        f"use_xla={use_xla}"
    )
    return config


class TFTRTConverter(BaseConverter):
    def __init__(
        self,
        *,
        is_dynamic_op: bool = False,
        minimum_segment_size: int = 3,
        max_batch_size: int = 1,
        max_workspace_size: int = (4 << 30) - 1000,  # ~3.999GB
        maximum_cached_engines: int = 1000,
        precision: str,
    ):
        self._is_dynamic_op = is_dynamic_op
        self._minimum_segment_size = minimum_segment_size
        self._max_batch_size = max_batch_size
        self._max_workspace_size = max_workspace_size
        self._maximum_cached_engines = maximum_cached_engines
        self._precision = Precision(precision)

    def convert(self, model: Model, dataloader_fn) -> Model:
        # https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
        # converting graph_def is not supported in TF2
        from tensorflow.python.compiler.tensorrt import trt_convert  # pytype: disable=import-error

        assert isinstance(model.handle, tf.compat.v1.GraphDef)

        session_config = create_session_config(allow_growth=True)
        output_node_names = [spec.name.split(":")[0] for spec in model.outputs.values()]

        converter = trt_convert.TrtGraphConverter(
            input_graph_def=model.handle,
            session_config=session_config,
            nodes_blacklist=output_node_names,
            is_dynamic_op=self._is_dynamic_op,
            precision_mode=self._precision.value,
            max_workspace_size_bytes=self._max_workspace_size,
            maximum_cached_engines=self._maximum_cached_engines,
            max_batch_size=self._max_batch_size,
            minimum_segment_size=self._minimum_segment_size,
        )
        graph_def = converter.convert()

        return model._replace(handle=graph_def)

    @staticmethod
    def required_source_model_precision(requested_model_precision: Precision) -> Precision:
        # TensorRT requires source models to be in FP32 precision
        return Precision.FP32


def _from_saved_model_v1(sess, model_path, tag, signatures):
    """
    Load tensorflow graph from saved_model.
    NOTICE: Modified version from tf2onnx project
    """

    wrn_no_tag = "'--tag' not specified for saved_model. Using --tag serve"
    wrn_empty_tag = "'--tag' value is empty string. Using tag =[[]]"

    if tag is None:
        tag = [tf.saved_model.SERVING]
        LOGGER.warning(wrn_no_tag)

    if tag == "":
        tag = [[]]
        LOGGER.warning(wrn_empty_tag)

    if not isinstance(tag, list):
        tag = [tag]

    imported = tf.compat.v1.saved_model.loader.load(sess, tag, model_path)
    for k in imported.signature_def.keys():
        if k.startswith("_"):
            # consider signatures starting with '_' private
            continue
        signatures.append(k)
    try:
        from tensorflow.contrib.saved_model.python.saved_model import (  # pytype: disable=import-error
            signature_def_utils,
        )

        # pylint: disable=unnecessary-lambda
        get_signature_def = lambda meta_graph_def, k: signature_def_utils.get_signature_def_by_key(meta_graph_def, k)
    except ImportError:
        # TF1.12 changed the api
        get_signature_def = lambda meta_graph_def, k: meta_graph_def.signature_def[k]

    inputs = {}
    outputs = {}
    for k in signatures:
        inputs_tensor_info = get_signature_def(imported, k).inputs
        for name, input_tensor in inputs_tensor_info.items():
            inputs[name] = input_tensor.name
        outputs_tensor_info = get_signature_def(imported, k).outputs
        for name, output_tensor in outputs_tensor_info.items():
            outputs[name] = output_tensor.name
    frozen_graph = freeze_session(sess, input_names=list(inputs.values()), output_names=list(outputs.values()))
    return frozen_graph, inputs, outputs


def _infer_model_precision(
    tf_graph: tf.compat.v1.GraphDef, inputs_dict: Dict[str, TensorSpec], outputs_dict: Dict[str, TensorSpec]
) -> Optional[Precision]:
    import networkx as nx

    def _get_dtype(node_def):
        node_type = node_def.attr.get("T", None) or node_def.attr.get("dtype", None)
        if node_type:
            if node_type.list.type:
                assert len(set(node_type.list.type)) == 1
                node_type = tf.dtypes.DType(node_type.list.type[0])
            else:
                node_type = tf.dtypes.DType(node_type.type)
        return np.dtype(node_type.as_numpy_dtype()) if node_type and node_type.is_numpy_compatible else node_type

    # build directed graph
    nx_graph = nx.DiGraph()
    for node_def in tf_graph.node:
        nx_graph.add_node(
            node_def.name,
            op=node_def.op,
            **{key: value for key, value in node_def.attr.items() if key not in ["value", "dtype"]},
            dtype=_get_dtype(node_def),
        )
        for input in node_def.input:
            nx_graph.add_edge(input, node_def.name)

    input_names = [spec.name.split(":")[0] for spec in inputs_dict.values()]
    output_names = [spec.name.split(":")[0] for spec in outputs_dict.values()]
    most_common_dtype = infer_precision(nx_graph, input_names, output_names, _get_dtype)
    if most_common_dtype is not None:
        precision = {np.dtype("float32"): Precision.FP32, np.dtype("float16"): Precision.FP16}[most_common_dtype]
    else:
        precision = None
    return precision


class TFEstimatorLoader(BaseLoader):
    required_fn_name_for_signature_parsing: Optional[str] = GET_MODEL_FN_NAME

    def __init__(self, **kwargs):
        self._model_args = kwargs

    def load(self, model_path: Union[str, Path], **_) -> Model:
        if isinstance(model_path, Path):
            model_path = model_path.as_posix()

        get_model = load_from_file(model_path, "model", GET_MODEL_FN_NAME)
        get_serving_input_receiver_fn = load_from_file(model_path, "model", GET_SERVING_INPUT_RECEIVER_FN)

        if get_model is None:
            raise RuntimeError(f"Could not find {GET_MODEL_FN_NAME} in {model_path}")
        if get_serving_input_receiver_fn is None:
            raise RuntimeError(f"Could not find {GET_SERVING_INPUT_RECEIVER_FN} in {model_path}")

        model_args = filter_fn_args(self._model_args, fn=get_model)
        serving_input_receiver_args = filter_fn_args(self._model_args, fn=get_serving_input_receiver_fn)

        session_config = create_session_config(allow_growth=True)
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session(config=session_config) as sess:
            estimator = get_model(**model_args)
            serving_input_receiver_fn = get_serving_input_receiver_fn(**serving_input_receiver_args)

            input_receiver = serving_input_receiver_fn()
            estimator_spec = estimator.model_fn(
                features=input_receiver.features,
                labels=None,
                mode=tf.estimator.ModeKeys.PREDICT,
                config=estimator.config,
            )

            input_tensors_dict = input_receiver.receiver_tensors
            output_tensors_dict = estimator_spec.predictions
            inputs_dict = {k: tensor2tensor_spec(tensor) for k, tensor in input_tensors_dict.items()}
            outputs_dict = {k: tensor2tensor_spec(tensor) for k, tensor in output_tensors_dict.items()}

            input_tensor_names = [t.name for t in inputs_dict.values()]
            output_tensor_names = [t.name for t in outputs_dict.values()]

            graph_saver = estimator_spec.scaffold.saver or tf.compat.v1.train.Saver(sharded=True)
            graph_saver.restore(sess, estimator.latest_checkpoint())

            input_tensor_names = inputs_without_resource(sess, input_tensor_names)
            frozen_graph = freeze_session(sess, input_names=input_tensor_names, output_names=output_tensor_names)
            input_tensor_names = remove_redundant_inputs(frozen_graph, input_tensor_names)

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session(config=estimator.config.session_config):
            frozen_graph = tf_optimize(input_tensor_names, output_tensor_names, frozen_graph)
        tf.compat.v1.reset_default_graph()

        precision = _infer_model_precision(frozen_graph, inputs_dict, outputs_dict)

        return Model(frozen_graph, precision, inputs_dict, outputs_dict)


class TFKerasLoader(BaseLoader):
    """
    Loads keras model from source code

    The large-model flag helps loading model which exceeds maximum protobuf size of 2GB. By default it is disabled.

    The tf-allow-growth flag control limiting GPU memory growth feature
    (https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth). By default it is disabled.
    """

    required_fn_name_for_signature_parsing: Optional[str] = GET_MODEL_FN_NAME

    def __init__(self, large_model: bool = False, tf_allow_growth: bool = False, **kwargs):
        self._large_model = large_model
        self._allow_growth = tf_allow_growth
        self._model_args = kwargs

    def load(self, model_path: Union[str, Path], **_) -> Model:
        if isinstance(model_path, Path):
            model_path = model_path.as_posix()

        get_model = load_from_file(model_path, "model", GET_MODEL_FN_NAME)
        if get_model is None:
            raise RuntimeError(f"Could not find {GET_MODEL_FN_NAME} in {model_path}")

        model_args = filter_fn_args(self._model_args, fn=get_model)

        if self._allow_growth:
            physical_devices = tf.config.experimental.list_physical_devices("GPU")
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)

        tf.keras.backend.clear_session()
        tf.keras.backend.set_learning_phase(False)

        eager_model, call_fn = get_model(**model_args)

        inputs_dict: Dict[str, TensorSpec] = {
            input_name: TensorSpec(t.name, t.dtype.name, tuple(t.shape.as_list()))
            for input_name, t in zip(eager_model.input_names, eager_model.inputs)
        }

        concrete_func = call_fn.get_concrete_function(
            *[tf.TensorSpec(shape=spec.shape, dtype=spec.dtype, name=name) for name, spec in inputs_dict.items()]
        )

        input_tensors_names = [tensor.name for tensor in concrete_func.inputs if tensor.dtype != tf.dtypes.resource]
        output_tensors_names = [tensor.name for tensor in concrete_func.outputs]

        graph_def = from_function(
            concrete_func, input_tensors_names, output_tensors_names, large_model=self._large_model
        )

        # tensor names changes after wrapping with call_fn, thus need to use those from concrete_func
        outputs_dict: Dict[str, TensorSpec] = {
            output_name: TensorSpec(output_tensor_name, t.dtype.name, tuple(t.shape.as_list()))
            for output_name, output_tensor_name, t in zip(
                eager_model.output_names, output_tensors_names, eager_model.outputs
            )
        }

        precision = _infer_model_precision(graph_def, inputs_dict, outputs_dict)

        tf.keras.backend.clear_session()
        tf.keras.backend.set_learning_phase(False)

        def _add_suffix_as_quickfix_for_tf24_func_refactor(spec):
            if not spec.name.endswith(":0"):
                spec = spec._replace(name=spec.name + ":0")
            return spec

        inputs_dict = {name: _add_suffix_as_quickfix_for_tf24_func_refactor(spec) for name, spec in inputs_dict.items()}

        return Model(graph_def, precision, inputs_dict, outputs_dict)


class TFSavedModelLoader(BaseLoader):
    def load(self, model_path: Union[str, Path], **kwargs) -> Model:
        if isinstance(model_path, Path):
            model_path = model_path.as_posix()
        tf.compat.v1.reset_default_graph()
        if is_tf2():
            from tf2onnx.tf_loader import _from_saved_model_v2  # pytype: disable=import-error

            graph_def, input_names, output_names, concrete_func, imported, initialized_tables = _from_saved_model_v2(
                model_path=model_path,
                input_names=None,
                output_names=None,
                tag=None,
                signature_def=[],
                concrete_function_index=None,
                large_model=False,
            )

            # inspired by https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/saved_model_cli.py#L205
            if concrete_func.structured_input_signature:
                input_args, input_kwargs = concrete_func.structured_input_signature
                input_names = list(input_kwargs)
                assert (
                    not input_args
                ), f"Not supported args in concrete function signature args={input_args}, kwargs={input_kwargs}"
            elif concrete_func._arg_keywords:  # pylint: disable=protected-access
                # For pure ConcreteFunctions we might have nothing better than _arg_keywords.
                assert concrete_func._num_positional_args in [0, 1]
                input_names = concrete_func._arg_keywords

            input_tensors = [tensor for tensor in concrete_func.inputs if tensor.dtype != tf.dtypes.resource]
            inputs = {name: tensor.name for name, tensor in zip(input_names, input_tensors)}

            # they are already flattened
            output_tensors = [tensor for tensor in concrete_func.outputs if tensor.dtype != tf.dtypes.resource]
            output_names = sorted(concrete_func.structured_outputs)  # because outputs are in flatten form
            outputs = {name: tensor.name for name, tensor in zip(output_names, output_tensors)}
        else:
            session_config = create_session_config(allow_growth=True)
            with tf.compat.v1.Session(config=session_config) as sess:
                graph_def, inputs, outputs = _from_saved_model_v1(sess, model_path, tag=None, signatures=[])

        inputs, outputs = handle_tensor_specs(graph_def, inputs, outputs)

        precision = _infer_model_precision(graph_def, inputs, outputs)

        return Model(graph_def, precision, inputs, outputs)


class TFRunner(BaseRunner):
    def __init__(self):
        pass

    def init_inference(self, model: Model):
        if is_tf2():
            return TF2RunnerSession(model=model)
        else:
            return TF1RunnerSession(model=model)


class TF1RunnerSession(BaseRunnerSession):
    def __init__(self, model: Model):
        super().__init__(model)

        assert isinstance(model.handle, tf.compat.v1.GraphDef)

        self._inputs = None
        self._outputs = None
        self._session = None
        self._old_env_values = {}

    def __enter__(self):
        self._old_env_values = self._set_env_variables()

        tf.compat.v1.reset_default_graph()

        session_config = create_session_config(allow_growth=True)
        self._session = tf.compat.v1.Session(config=session_config)
        self._session.__enter__()

        tf.import_graph_def(self._model.handle, name="")

        self._inputs = {
            name: self._session.graph.get_tensor_by_name(spec.name) for name, spec in self._model.inputs.items()
        }
        self._outputs = {
            name: self._session.graph.get_tensor_by_name(spec.name) for name, spec in self._model.outputs.items()
        }
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._session.__exit__(exc_type, exc_value, traceback)
        tf.compat.v1.reset_default_graph()
        self._inputs = None
        self._outputs = None
        self._session = None
        self._recover_env_variables(self._old_env_values)

    def __call__(self, x: Dict[str, object]):
        feed_dict = {placeholder: x[name] for name, placeholder in self._inputs.items()}
        return self._session.run(self._outputs, feed_dict=feed_dict)


class TF2RunnerSession(BaseRunnerSession):
    def __init__(self, model: Model):
        super().__init__(model)
        assert isinstance(model.handle, tf.compat.v1.GraphDef)
        self._concrete_func = None

    def __enter__(self):
        tf.compat.v1.reset_default_graph()
        input_tensor_names = [spec.name for spec in self._model.inputs.values()]
        output_tensor_names = [spec.name for spec in self._model.outputs.values()]
        self._concrete_func = wrap_function.function_from_graph_def(
            self._model.handle, input_tensor_names, output_tensor_names
        )
        self._concrete_func._signature = [
            tf.TensorSpec(shape=spec.shape, dtype=spec.dtype, name=name) for name, spec in self._model.inputs.items()
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._concrete_func = None
        tf.compat.v1.reset_default_graph()

    def __call__(self, x: Dict[str, object]):
        x = tf.nest.map_structure(tf.convert_to_tensor, x)
        y_pred = self._concrete_func(**x)
        output_struct = {name: spec.name for name, spec in self._model.outputs.items()}
        y_pred = tf.nest.map_structure(lambda t: t.numpy(), y_pred)
        y_pred = tf.nest.pack_sequence_as(output_struct, y_pred)
        return y_pred


class TFSavedModelSaver(BaseSaver):
    def save(self, model: Model, model_path: Union[str, Path]) -> None:
        if isinstance(model_path, Path):
            model_path = model_path.as_posix()

        session_config = create_session_config(allow_growth=True)
        with tf.compat.v1.Session(config=session_config) as sess:
            tf.import_graph_def(model.handle, name="")

            is_func = is_function(sess.graph)
            if not is_func:
                infer_shape(sess.graph, {})

            inputs = {name: sess.graph.get_tensor_by_name(spec.name) for name, spec in model.inputs.items()}
            outputs = {name: sess.graph.get_tensor_by_name(spec.name) for name, spec in model.outputs.items()}

            def _ensure_shape(tensors_dict, tensors_specs):
                for name, tensor in tensors_dict.items():
                    if tensor.shape.rank is None:
                        tensor.set_shape(tensors_specs[name].shape)
                return tensors_dict

            inputs = _ensure_shape(inputs, model.inputs)
            outputs = _ensure_shape(outputs, model.outputs)

            LOGGER.info(inputs)
            LOGGER.info(outputs)

            tf.compat.v1.saved_model.simple_save(sess, model_path, inputs, outputs, legacy_init_op=None)


def handle_tensor_specs(
    graph_def, inputs: Dict[str, str], outputs: Dict[str, str]
) -> Tuple[Dict[str, TensorSpec], Dict[str, TensorSpec]]:
    session_config = tf.compat.v1.ConfigProto(graph_options=tf.compat.v1.GraphOptions(infer_shapes=True))
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session(config=session_config) as sess:
        tf.import_graph_def(graph_def, name="")

        def _get_spec(tensors_dict):
            tensors_dict = {name: sess.graph.get_tensor_by_name(tname) for name, tname in tensors_dict.items()}
            return {name: tensor2tensor_spec(tensor) for name, tensor in tensors_dict.items()}

        inputs = _get_spec(inputs)
        outputs = _get_spec(outputs)

    tf.compat.v1.reset_default_graph()
    return inputs, outputs


def tensor2tensor_spec(tensor):
    shape = tuple([s.value if hasattr(s, "value") else s for s in tensor.shape])
    return TensorSpec(tensor.name, tensor.dtype.name, shape)


loaders.register_extension(Format.TF_ESTIMATOR.value, TFEstimatorLoader)
loaders.register_extension(Format.TF_KERAS.value, TFKerasLoader)
loaders.register_extension(Format.TF_SAVEDMODEL.value, TFSavedModelLoader)
loaders.register_extension(Format.TF_TRT.value, TFSavedModelLoader)

converters.register_extension(f"{Format.TF_ESTIMATOR.value}--{Format.TF_SAVEDMODEL.value}", None)
converters.register_extension(f"{Format.TF_KERAS.value}--{Format.TF_SAVEDMODEL.value}", None)
converters.register_extension(f"{Format.TF_SAVEDMODEL.value}--{Format.TF_SAVEDMODEL.value}", None)
converters.register_extension(f"{Format.TF_ESTIMATOR.value}--{Format.TF_TRT.value}", TFTRTConverter)
converters.register_extension(f"{Format.TF_KERAS.value}--{Format.TF_TRT.value}", TFTRTConverter)
converters.register_extension(f"{Format.TF_SAVEDMODEL.value}--{Format.TF_TRT.value}", TFTRTConverter)

savers.register_extension(Format.TF_SAVEDMODEL.value, TFSavedModelSaver)
savers.register_extension(Format.TF_TRT.value, TFSavedModelSaver)

runners.register_extension(Format.TF_ESTIMATOR.value, TFRunner)
runners.register_extension(Format.TF_KERAS.value, TFRunner)
runners.register_extension(Format.TF_SAVEDMODEL.value, TFRunner)
runners.register_extension(Format.TF_TRT.value, TFRunner)
