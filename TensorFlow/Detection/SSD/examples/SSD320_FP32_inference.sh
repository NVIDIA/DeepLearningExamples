PIPELINE_CONFIG_PATH=/workdir/models/research/configs/ssd320_full_1gpus.config

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
OBJECT_DETECTION=$(realpath $SCRIPT_DIR/../object_detection/)
PYTHONPATH=$PYTHONPATH:$OBJECT_DETECTION

python $SCRIPT_DIR/SSD320_inference.py \
       --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
       "$@"
