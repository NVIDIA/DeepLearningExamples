PIPELINE_CONFIG_PATH=/workdir/models/research/configs/ssd320_full_8gpus.config
CKPT_DIR=${1:-"/results/SSD320_FP16_8GPU"}
GPUS=8

export TF_ENABLE_AUTO_MIXED_PRECISION=1

TENSOR_OPS=0
export TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32=${TENSOR_OPS}
export TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32=${TENSOR_OPS}
export TF_ENABLE_CUDNN_RNN_TENSOR_OP_MATH_FP32=${TENSOR_OPS}

time mpirun --allow-run-as-root \
       -np $GPUS \
       -H localhost:$GPUS \
       -bind-to none \
       -map-by slot \
       -x NCCL_DEBUG=INFO \
       -x LD_LIBRARY_PATH \
       -x PATH \
       -mca pml ob1 \
       -mca btl ^openib \
        python -u /workdir/models/research/object_detection/model_main.py \
               --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
               --model_dir=${CKPT_DIR} \
               --alsologtostder \
               "${@:2}"
