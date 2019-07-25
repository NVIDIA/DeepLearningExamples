#! /bin/bash

mpiexec --allow-run-as-root --bind-to socket -np 8 python3 run_pretraining.py \
  --input_file=/workspace/data/bert_large_wikipedia_seq_512_pred_20/tf_examples.tfrecord* \
  --output_dir=/workspace/checkpoints/pretraining_base_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=14 \
  --max_seq_length=512 \
  --max_predictions_per_seq=20 \
  --num_train_steps=250000 \
  --num_warmup_steps=10000 \
  --learning_rate=1e-4 \
  --use_fp16 \
  --use_xla \
  --report_loss \
  --horovod

