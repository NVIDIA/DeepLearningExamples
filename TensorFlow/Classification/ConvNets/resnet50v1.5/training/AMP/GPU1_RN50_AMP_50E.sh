WORKSPACE=${1:-"/workspace/rn50v15_tf"}
DATA_DIR=${2:-"/data"}

bash ${WORKSPACE}/resnet50v1.5/training/GENERIC.sh ${WORKSPACE} ${DATA_DIR} \
    1 50 256 amp 
