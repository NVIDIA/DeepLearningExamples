# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

readonly num_shards_per_worker=128 # num_shards_per_worker * num_workers * num_gpus * num_nodes == 4096 seems to be a good number.
readonly num_workers=4
readonly num_nodes=1
readonly num_gpus=8
readonly phase2_bin_size=none # none or 32 or 64 or 128 or 256 or 512
readonly masking=static # static or dynamic
readonly num_dask_workers=128 # 128 on dgxa100, 64 on dgx2 and dgx1
readonly VOCAB_FILE=vocab/vocab
readonly sample_ratio=0.9 # to match the original

readonly DATA_DIR_PHASE1=${BERT_PREP_WORKING_DIR}/pretrain/phase1/parquet/
readonly DATA_DIR_PHASE2=${BERT_PREP_WORKING_DIR}/pretrain/phase2/parquet/
readonly wikipedia_source=${BERT_PREP_WORKING_DIR}/wikipedia/source/

# Calculate the total number of shards.
readonly num_blocks=$((num_shards_per_worker * $(( num_workers > 0 ? num_workers : 1 )) * num_nodes * num_gpus))

if [ "${phase2_bin_size}" == "none" ]; then
   readonly phase2_bin_size_flag=""
elif [[ "${phase2_bin_size}" =~ ^(32|64|128|256|512)$ ]]; then
   readonly phase2_bin_size_flag="--bin-size ${phase2_bin_size}"
else
   echo "Error! phase2_bin_size=${phase2_bin_size} not supported!"
   return 1
fi

if [ "${masking}" == "static" ]; then
   readonly masking_flag="--masking"
elif [ "${masking}" == "dynamic" ]; then
   readonly masking_flag=""
else
   echo "Error! masking=${masking} not supported!"
   return 1
fi

# Run the preprocess pipeline for phase 1.
if [ ! -d "${DATA_DIR_PHASE1}" ] || [ -z "$(ls -A ${DATA_DIR_PHASE1})" ]; then
   echo "Warning! ${DATA_DIR_PHASE1} directory missing."
   if [ ! -d "${wikipedia_source}" ] || [ -z "$(ls -A ${wikipedia_source})" ]; then
      echo "Error! ${wikipedia_source} directory missing. Training cannot start!"
      return 1
   fi
   preprocess_cmd=" \
      mpirun \
         --oversubscribe \
         --allow-run-as-root \
         -np ${num_dask_workers} \
         -x LD_PRELOAD=/opt/conda/lib/libjemalloc.so \
            preprocess_bert_pretrain \
               --schedule mpi \
               --vocab-file ${VOCAB_FILE} \
               --wikipedia ${wikipedia_source} \
               --sink ${DATA_DIR_PHASE1} \
               --num-blocks ${num_blocks} \
               --sample-ratio ${sample_ratio} \
               ${masking_flag} \
               --seed ${seed}"
   echo "Running ${preprocess_cmd} ..."
   ${preprocess_cmd}

   balance_load_cmd=" \
      mpirun \
         --oversubscribe \
         --allow-run-as-root \
         -np ${num_dask_workers} \
            balance_dask_output \
               --indir ${DATA_DIR_PHASE1} \
               --num-shards ${num_blocks}"
   echo "Running ${balance_load_cmd} ..."
   ${balance_load_cmd}
fi

#Distillation phase1
python -m torch.distributed.launch --nproc_per_node=${num_gpus} general_distill.py \
  --input_dir ${DATA_DIR_PHASE1} \
  --teacher_model checkpoints/bert-base-uncased \
  --student_model BERT_4L_312D \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 1e-4 \
  --output_dir checkpoints/nv_distill_p1 \
  --distill_config distillation_config_backbone.json \
  --seed=1 \
  --amp \
  --vocab_file ${VOCAB_FILE} \
  --num_workers ${num_workers} | tee -a test_nv_distill_p1.log

# Run the preprocess pipeline for phase 2.
if [ ! -d "${DATA_DIR_PHASE2}" ] || [ -z "$(ls -A ${DATA_DIR_PHASE2})" ]; then
   echo "Warning! ${DATA_DIR_PHASE2} directory missing."
   if [ ! -d "${wikipedia_source}" ] || [ -z "$(ls -A ${wikipedia_source})" ]; then
      echo "Error! ${wikipedia_source} directory missing. Training cannot start!"
      return 1
   fi
   preprocess_cmd=" \
      mpirun \
         --oversubscribe \
         --allow-run-as-root \
         -np ${num_dask_workers} \
         -x LD_PRELOAD=/opt/conda/lib/libjemalloc.so \
            preprocess_bert_pretrain \
               --schedule mpi \
               --vocab-file ${VOCAB_FILE} \
               --wikipedia ${wikipedia_source} \
               --sink ${DATA_DIR_PHASE2} \
               --target-seq-length 512 \
               --num-blocks ${num_blocks} \
               --sample-ratio ${sample_ratio} \
               ${phase2_bin_size_flag} \
               ${masking_flag} \
               --seed ${seed}"
   echo "Running ${preprocess_cmd} ..."
   ${preprocess_cmd}

   balance_load_cmd=" \
      mpirun \
         --oversubscribe \
         --allow-run-as-root \
         -np ${num_dask_workers} \
            balance_dask_output \
               --indir ${DATA_DIR_PHASE2} \
               --num-shards ${num_blocks}"
   echo "Running ${balance_load_cmd} ..."
   ${balance_load_cmd}
fi

#Distillation phase2
python -m torch.distributed.launch --nproc_per_node=${num_gpus} general_distill.py
  --input_dir ${DATA_DIR_PHASE2} \
  --teacher_model checkpoints/bert-base-uncased \
  --student_model checkpoints/nv_distill_p1 \
  --do_lower_case \
  --train_batch_size 32 \
  --learning_rate 1e-4 \
  --output_dir checkpoints/nv_distill_p2 \
  --distill_config distillation_config_backbone.json \
  --seed=1 \
  --max_steps 6861 \
  --steps_per_epoch 2287 \
  --max_seq_length 512 \
  --continue_train \
  --amp \
  --vocab_file ${VOCAB_FILE} \
  --num_workers ${num_workers} | tee -a test_nv_distill_p2.log

#Distillation SQUAD

#Data aug
python data_augmentation.py --pretrained_bert_model checkpoints/bert-base-uncased --glove_embs $BERT_PREP_WORKING_DIR/download/glove/glove.6B.300d.txt --glue_dir $BERT_PREP_WORKING_DIR/download/squad/v1.1 --task_name SQuADv1.1 --seed=1

# backbone loss
python -m torch.distributed.launch --nproc_per_node=8 task_distill.py --teacher_model checkpoints/bert-base-uncased-qa --student_model checkpoints/nv_distill_p2 --data_dir $BERT_PREP_WORKING_DIR/download/squad/v1.1 --task_name SQuADv1.1 --output_dir checkpoints/nv_distill_squad --max_seq_length 384 --train_batch_size 32 --num_train_epochs 9 --do_lower_case --distill_config distillation_config_backbone.json --aug_train --amp --seed=1 | tee -a test_nv_distill_squad.log

# prediction loss
python -m torch.distributed.launch --nproc_per_node=8 task_distill.py --teacher_model checkpoints/bert-base-uncased-qa --student_model checkpoints/nv_distill_squad --data_dir $BERT_PREP_WORKING_DIR/download/squad/v1.1 --task_name SQuADv1.1 --output_dir checkpoints/nv_distill_squad_pred --max_seq_length 384 --train_batch_size 32 --num_train_epochs 10 --do_lower_case --distill_config distillation_config_heads.json --aug_train --learning_rate 3e-5 --eval_step 2000 --amp --seed=1 | tee -a test_nv_distill_squad_pred.log

#Distillation SST

python data_augmentation.py --pretrained_bert_model checkpoints/bert-base-uncased --glove_embs $BERT_PREP_WORKING_DIR/download/glove/glove.6B.300d.txt --glue_dir $BERT_PREP_WORKING_DIR/download/glue --task_name SST-2 --seed=1

#backbone loss
python -m torch.distributed.launch --nproc_per_node=8 task_distill.py \
--teacher_model checkpoints/bert-base-uncased-sst-2 \
--student_model checkpoints/nv_distill_p1 \
--data_dir $BERT_PREP_WORKING_DIR/download/glue/SST-2 \
--task_name SST-2 --output_dir checkpoints/nv_distill_sst_2 --max_seq_length 128 \
--train_batch_size 32  --num_train_epochs 10  --do_lower_case --aug_train --amp --distill_config distillation_config_backbone.json \
                                --seed=1 \
 | tee -a test_nv_distill_sst_2.log; \

# prediciton loss
python -m torch.distributed.launch --nproc_per_node=8 task_distill.py \
--teacher_model checkpoints/bert-base-uncased-sst-2 \
--student_model checkpoints/nv_distill_sst_2 \
--data_dir $BERT_PREP_WORKING_DIR/download/glue/SST-2 \
--task_name SST-2 --output_dir checkpoints/nv_distill_sst_2_pred --max_seq_length 128 \
--train_batch_size 32  --num_train_epochs 3  --do_lower_case --aug_train --amp --distill_config distillation_config_heads.json \
                                --seed=1 --learning_rate 3e-5 --eval_step 100 \
 | tee -a test_nv_distill_sst_2_pred.log
