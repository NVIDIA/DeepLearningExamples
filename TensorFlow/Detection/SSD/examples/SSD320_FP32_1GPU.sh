PIPELINE_CONFIG_PATH=/workdir/models/research/configs/ssd320_full_1gpus.config
CKPT_DIR=${1:-"/results/SSD320_FP32_1GPU"}

TENSOR_OPS=0
export TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=${TENSOR_OPS}
export TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=${TENSOR_OPS}
export TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=${TENSOR_OPS}

time python -u /workdir/models/research/object_detection/model_main.py \
       --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
       --model_dir=${CKPT_DIR} \
       --alsologtostder \
       "${@:2}"
