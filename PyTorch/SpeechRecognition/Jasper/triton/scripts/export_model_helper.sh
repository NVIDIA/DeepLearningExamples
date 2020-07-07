#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SCRIPT_DIR=$(cd $(dirname $0); pwd)
#### input arguments
CHECKPOINT=${CHECKPOINT}
PRECISION=${PRECISION:-fp16}
ARCH=${ARCH:-75}
MODEL_REPO=${MODEL_REPO:-"${SCRIPT_DIR}/../model_repo"}
JASPER_REPO=${JASPER_REPO:-"${SCRIPT_DIR}/../.."}
MODEL_CONFIG=${MODEL_CONFIG:-"jasper10x5dr_nomask.toml"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/checkpoints"}
MAX_SEQUENCE_LENGTH_FOR_ENGINE=${MAX_SEQUENCE_LENGTH_FOR_ENGINE}
####



export PYTHONPATH=${JASPER_REPO}

echo "export_model.sh: Exporting TorchScript ... "

mkdir -p ${MODEL_REPO}/jasper-trt/1/
mkdir -p ${MODEL_REPO}/jasper-onnx/1/
mkdir -p ${MODEL_REPO}/jasper-pyt/1/
mkdir -p ${MODEL_REPO}/jasper-trt-ensemble/1/
mkdir -p ${MODEL_REPO}/jasper-onnx-ensemble/1/
mkdir -p ${MODEL_REPO}/jasper-pyt-ensemble/1/

mkdir -p ${MODEL_REPO}/jasper-feature-extractor/1/
mkdir -p ${MODEL_REPO}/jasper-decoder/1/

PREC_FLAGS=""
if [ "$PRECISION" == "fp16" ]
then
	PREC_FLAGS="--fp16 --pyt_fp16"
fi

python  ${JASPER_REPO}/inference.py \
	--ckpt ${CHECKPOINT_DIR}/${CHECKPOINT} \
	--wav=${JASPER_REPO}/notebooks/example1.wav  \
	--model_toml=${JASPER_REPO}/configs/${MODEL_CONFIG} \
	--export_model --output_dir ${PWD} ${PREC_FLAGS} ${ADDITIONAL_ARGS} || exit 1

mv *_feat.pt ${MODEL_REPO}/jasper-feature-extractor/1/jasper-feature-extractor.pt
mv *_acoustic.pt ${MODEL_REPO}/jasper-pyt/1/jasper.pt
mv *_decoder.pt ${MODEL_REPO}/jasper-decoder/1/jasper-decoder.pt

echo "TorchScript export succeeded."
echo "export_model.sh: Exporting ONNX and TRT ... "

# we need 2 separate export passes because OSS TRT ONNX parser currently chokes on hybrid ONNX
echo "export_model.sh: Exporting TRT engine, CUDA ARCH = ${ARCH} ... "

PREC_FLAGS=""
if [ "$PRECISION" == "fp16" ]
then
	PREC_FLAGS="--trt_fp16"
fi

# remove targtes first
rm -f ${MODEL_REPO}/jasper-trt/1/jasper_${ARCH}.plan ${MODEL_REPO}/jasper-onnx/1/jasper.onnx

python  ${JASPER_REPO}/trt/perf.py \
	--ckpt_path ${CHECKPOINT_DIR}/${CHECKPOINT} \
	--wav=${JASPER_REPO}/notebooks/example1.wav \
	--model_toml=${JASPER_REPO}/configs/${MODEL_CONFIG} \
	--make_onnx --onnx_path jasper-tmp.onnx --engine_path ${MODEL_REPO}/jasper-trt/1/jasper_${ARCH}.plan --seq_len=256 --max_seq_len ${MAX_SEQUENCE_LENGTH_FOR_ENGINE} --verbose ${PREC_FLAGS} || exit 1
rm -fr jasper-tmp.onnx


PREC_FLAGS=""
if [ "$PRECISION" == "fp16" ]
then
	PREC_FLAGS="--trt_fp16 --pyt_fp16"
fi
python  ${JASPER_REPO}/trt/perf.py \
	--ckpt_path ${CHECKPOINT_DIR}/${CHECKPOINT} \
	--wav=${JASPER_REPO}/notebooks/example1.wav \
	--model_toml=${JASPER_REPO}/configs/${MODEL_CONFIG} \
	--make_onnx --onnx_path ${MODEL_REPO}/jasper-onnx/1/jasper.onnx --seq_len=256 --max_seq_len ${MAX_SEQUENCE_LENGTH_FOR_ENGINE} --verbose ${PREC_FLAGS} ${ADDITIONAL_TRT_ARGS} || exit 1

mkdir -p ${MODEL_REPO}/jasper-onnx-cpu/1
cp -f ${MODEL_REPO}/jasper-onnx/1/jasper.onnx ${MODEL_REPO}/jasper-onnx-cpu/1/jasper.onnx 

if [ -f /.dockerenv ]; then # inside docker
    # Make sure we do not leave read-only root-owned files
    chmod -R a+wrX ${MODEL_REPO}/
fi

echo "export_model.sh: Exporting ONNX and TRT engines succeeded "
