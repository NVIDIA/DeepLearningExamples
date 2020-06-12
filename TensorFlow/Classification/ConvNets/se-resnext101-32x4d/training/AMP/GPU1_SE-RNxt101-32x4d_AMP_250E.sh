RESULT_DIR=${1:-"/workspace/rn50v15_tf/results"}
WORKSPACE=${2:-"/workspace/rn50v15_tf"}
DATA_DIR=${3:-"/data"}

bash ${WORKSPACE}/se-resnext101-32x4d/training/GENERIC.sh ${RESULT_DIR} ${DATA_DIR} \
    1 250 128 amp --mixup=0.2 
