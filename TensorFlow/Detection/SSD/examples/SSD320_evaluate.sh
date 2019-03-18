CHECKPINT_DIR=$1

TENSOR_OPS=0
export TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=${TENSOR_OPS}
export TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=${TENSOR_OPS}
export TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=${TENSOR_OPS}

python object_detection/model_main.py --checkpoint_dir $CHECKPINT_DIR --model_dir /results --run_once --pipeline_config_path configs/ssd320_full_1gpus.config
