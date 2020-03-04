# Usage: run_benchmark(batch_sizes, model_variant: (base/large), precision: (fp16/fp32), sequence_length, max_batch_size)
run_benchmark() {
BATCH_SIZES="${1}"

MODEL_VARIANT="${2}"
PRECISION="${3}"
SEQUENCE_LENGTH="${4}"
MAX_BATCH="${5}"

CHECKPOINTS_DIR="/workspace/bert/models/fine-tuned/bert_tf_v2_${MODEL_VARIANT}_${PRECISION}_${SEQUENCE_LENGTH}_v2"
ENGINE_NAME="/workspace/bert/engines/bert_${MODEL_VARIANT}_${PRECISION}_bs${MAX_BATCH}_seqlen${SEQUENCE_LENGTH}_benchmark.engine"

echo "==== Benchmarking BERT ${MODEL_VARIANT} ${PRECISION} SEQLEN ${SEQUENCE_LENGTH} ===="
if [ ! -f ${ENGINE_NAME} ]; then
    if [ ! -d ${CHECKPOINTS_DIR} ]; then
        echo "Downloading checkpoints: scripts/download_model.sh ${MODEL_VARIANT} ${PRECISION} ${SEQUENCE_LENGTH}"
        scripts/download_model.sh "${MODEL_VARIANT}" "${PRECISION}" "${SEQUENCE_LENGTH}"
    fi;

    echo "Building engine: python builder.py -m ${CHECKPOINTS_DIR}/model.ckpt-8144 -o ${ENGINE_NAME} ${BATCH_SIZES} -s ${SEQUENCE_LENGTH} --${PRECISION} -c ${CHECKPOINTS_DIR}"
    python builder.py -m ${CHECKPOINTS_DIR}/model.ckpt-8144 -o ${ENGINE_NAME} ${BATCH_SIZES} -s ${SEQUENCE_LENGTH} --${PRECISION} -c ${CHECKPOINTS_DIR}
fi;

python perf.py ${BATCH_SIZES} -s ${SEQUENCE_LENGTH} -e ${ENGINE_NAME}
echo
}

mkdir -p /workspace/bert/engines

# BERT BASE
## FP16
run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "base" "fp16" "128" "32"
run_benchmark "-b 64" "base" "fp16" "128" "64"
run_benchmark "-b 128" "base" "fp16" "128" "128"

run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "base" "fp16" "384" "32"
run_benchmark "-b 64" "base" "fp16" "384" "64"
run_benchmark "-b 128" "base" "fp16" "384" "128"

## FP32
run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "base" "fp32" "128" "32"
run_benchmark "-b 64" "base" "fp32" "128" "64"
run_benchmark "-b 128" "base" "fp32" "128" "128"

run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "base" "fp32" "384" "32"
run_benchmark "-b 64" "base" "fp32" "384" "64"
run_benchmark "-b 128" "base" "fp32" "384" "128"

# BERT LARGE
## FP16
run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "fp16" "128" "32"
run_benchmark "-b 64" "large" "fp16" "128" "64"
run_benchmark "-b 128" "large" "fp16" "128" "128"

run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "fp16" "384" "32"
run_benchmark "-b 64" "large" "fp16" "384" "64"
run_benchmark "-b 128" "large" "fp16" "384" "128"

## FP32
run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "fp32" "128" "32"
run_benchmark "-b 64" "large" "fp32" "128" "64"
run_benchmark "-b 128" "large" "fp32" "128" "128"

run_benchmark "-b 1 -b 2 -b 4 -b 8 -b 12 -b 16 -b 24 -b 32" "large" "fp32" "384" "32"
run_benchmark "-b 64" "large" "fp32" "384" "64"
run_benchmark "-b 128" "large" "fp32" "384" "128"
