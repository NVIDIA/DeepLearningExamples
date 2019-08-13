init_checkpoint=${1:-"/results/model.ckpt"}
batch_size=${2:-"8"}
precision=${3:-"fp16"}
use_xla=${4:-"true"}
seq_length=${5:-"384"}
doc_stride=${6:-"128"}
BERT_DIR=${7:-"data/pretrained_models_google/uncased_L-24_H-1024_A-16"}
trtis_model_version=${8:-1}
trtis_model_name=${9:-"bert"}
trtis_dyn_batching_delay=${10:-0}
trtis_engine_count=${11:-1}
trtis_model_overwrite=${12:-"False"}

additional_args="--trtis_model_version=$trtis_model_version --trtis_model_name=$trtis_model_name --trtis_max_batch_size=$batch_size \
                 --trtis_model_overwrite=$trtis_model_overwrite --trtis_dyn_batching_delay=$trtis_dyn_batching_delay \
                 --trtis_engine_count=$trtis_engine_count"

if [ "$precision" = "fp16" ] ; then
   echo "fp16 activated!"
   export TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=1
   additional_args="$additional_args --use_fp16"
fi

if [ "$use_xla" = "true" ] ; then
    echo "XLA activated"
    additional_args="$additional_args --use_xla"
fi

echo "Additional args: $additional_args"

bash scripts/docker/launch.sh \
    python run_squad.py \
       --vocab_file=${BERT_DIR}/vocab.txt \
       --bert_config_file=${BERT_DIR}/bert_config.json \
       --init_checkpoint=${init_checkpoint} \
       --max_seq_length=${seq_length} \
       --doc_stride=${doc_stride} \
       --predict_batch_size=${batch_size} \
       --output_dir=/results \
       --export_trtis=True \
       ${additional_args}



