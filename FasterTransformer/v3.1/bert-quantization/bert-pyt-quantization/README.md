# PyTorch BERT Quantization Example

Based on `https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT`

Original README [here](README_orig.md)

Modified the following files:
 * modeling.py
 * run_squad.py

## Setup

Please follow the original README to do some inital setup.

setup steps:
```bash
export ROOT_DIR=</path/to/this_repo_root_dir>
export DATA_DIR=</path/to/data_dir>
export MODEL_DIR=</path/to/model/checkpoint>

git clone https://github.com/NVIDIA/TensorRT.git
cd TensorRT
git checkout release/7.2
pip install tools/pytorch-quantization/.
```

download SQuAD data:
```bash
cd $DATA_DIR
bash $ROOT_DIR/data/squad/squad_download.sh
```

download pre-trained checkpoint, config file, and vocab file (bert-base-uncased):
```bash
cd $MODEL_DIR
mkdir bert-base-uncased
wget https://s3.amazonaws.com/models.huggingface.co/bert/google/bert_uncased_L-12_H-768_A-12/pytorch_model.bin -O bert-base-uncased/pytorch_model.bin
wget https://s3.amazonaws.com/models.huggingface.co/bert/google/bert_uncased_L-12_H-768_A-12/config.json -O bert-base-uncased/config.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/google/bert_uncased_L-12_H-768_A-12/vocab.txt -O bert-base-uncased/vocab.txt

cd $ROOT_DIR
```

## Post Training Quantization

Firstly, finetune for a float dense model:

```bash
python -m torch.distributed.launch --nproc_per_node=8 run_squad.py \
  --init_checkpoint=$MODEL_DIR/bert-base-uncased/pytorch_model.bin \
  --do_train \
  --train_file=$DATA_DIR/v1.1/train-v1.1.json \
  --train_batch_size=4 \
  --learning_rate=3e-5 \
  --num_train_epochs=2 \
  --do_predict \
  --predict_file=$DATA_DIR/v1.1/dev-v1.1.json \
  --eval_script=$DATA_DIR/v1.1/evaluate-v1.1.py \
  --do_eval \
  --do_lower_case \
  --bert_model=bert-base-uncased \
  --max_seq_length=384 \
  --doc_stride=128 \
  --vocab_file=$MODEL_DIR/bert-base-uncased/vocab.txt \
  --config_file=$MODEL_DIR/bert-base-uncased/config.json \
  --json-summary=$MODEL_DIR/dllogger.json \
  --output_dir=$MODEL_DIR/bert-base-uncased-finetuned \
  --max_steps=-1 \
  --fp16 \
  --quant_disable
```

The results would be like:

```bash
{"exact_match": 82.63, "f1": 89.53}
```

Then do PTQ, `ft_mode` is unified with int8_mode in FasterTransformer, can be one of `1` or `2`.

```bash
python run_squad.py \
  --init_checkpoint=$MODEL_DIR/bert-base-uncased-finetuned/pytorch_model.bin \
  --do_calib \
  --train_file=$DATA_DIR/v1.1/train-v1.1.json \
  --train_batch_size=16 \
  --num-calib-batch=16 \
  --do_predict \
  --predict_file=$DATA_DIR/v1.1/dev-v1.1.json \
  --eval_script=$DATA_DIR/v1.1/evaluate-v1.1.py \
  --do_eval \
  --do_lower_case \
  --bert_model=bert-base-uncased \
  --max_seq_length=384 \
  --doc_stride=128 \
  --vocab_file=$MODEL_DIR/bert-base-uncased/vocab.txt \
  --config_file=$MODEL_DIR/bert-base-uncased/config.json \
  --json-summary=$MODEL_DIR/dllogger.json \
  --output_dir=$MODEL_DIR/bert-base-uncased-PTQ-mode-2 \
  --max_steps=-1 \
  --fp16 \
  --calibrator percentile \
  --percentile 99.999 \
  --ft_mode 2
```

The results would be like:

```bash
{"exact_match": 81.93, "f1": 89.05}     # for mode 1
{"exact_match": 80.41, "f1": 88.15}     # for mode 2
```


## Quantization Aware Fine-tuning

If PTQ does not yield an acceptable result you can finetune with quantization to recover accuracy.
We recommend to calibrate the pretrained model and finetune to avoid overfitting:

`ft_mode` is unified with int8_mode in FasterTransformer, can be one of `1` or `2`.

```bash
python run_squad.py \
  --init_checkpoint=$MODEL_DIR/bert-base-uncased/pytorch_model.bin \
  --do_calib \
  --train_file=$DATA_DIR/v1.1/train-v1.1.json \
  --train_batch_size=16 \
  --num-calib-batch=16 \
  --do_lower_case \
  --bert_model=bert-base-uncased \
  --max_seq_length=384 \
  --doc_stride=128 \
  --vocab_file=$MODEL_DIR/bert-base-uncased/vocab.txt \
  --config_file=$MODEL_DIR/bert-base-uncased/config.json \
  --json-summary=$MODEL_DIR/dllogger.json \
  --output_dir=$MODEL_DIR/bert-base-uncased-calib-mode-2 \
  --max_steps=-1 \
  --fp16 \
  --calibrator percentile \
  --percentile 99.99 \
  --ft_mode 2

python -m torch.distributed.launch --nproc_per_node=8 run_squad.py \
  --init_checkpoint=$MODEL_DIR/bert-base-uncased-calib-mode-2/pytorch_model.bin \
  --do_train \
  --train_file=$DATA_DIR/v1.1/train-v1.1.json \
  --train_batch_size=4 \
  --learning_rate=3e-5 \
  --num_train_epochs=2 \
  --do_predict \
  --predict_file=$DATA_DIR/v1.1/dev-v1.1.json \
  --eval_script=$DATA_DIR/v1.1/evaluate-v1.1.py \
  --do_eval \
  --do_lower_case \
  --bert_model=bert-base-uncased \
  --max_seq_length=384 \
  --doc_stride=128 \
  --vocab_file=$MODEL_DIR/bert-base-uncased/vocab.txt \
  --config_file=$MODEL_DIR/bert-base-uncased/config.json \
  --json-summary=$MODEL_DIR/dllogger.json \
  --output_dir=$MODEL_DIR/bert-base-uncased-QAT-mode-2 \
  --max_steps=-1 \
  --fp16 \
  --ft_mode 2
```

The results would be like:

```bash
{"exact_match": 81.91, "f1": 89.09}     # for mode 1
{"exact_match": 81.72, "f1": 89.09}     # for mode 2
```

The results of quantization may differ if different seeds are provided.


## Quantization Aware Fine-tuning with Knowledge-distillation

Knowledge-distillation can get better results, we usually starts from a PTQ checkpoint.

```bash
python run_squad.py \
  --init_checkpoint=$MODEL_DIR/bert-base-uncased-finetuned/pytorch_model.bin \
  --do_calib \
  --train_file=$DATA_DIR/v1.1/train-v1.1.json \
  --train_batch_size=16 \
  --num-calib-batch=16 \
  --do_lower_case \
  --bert_model=bert-base-uncased \
  --max_seq_length=384 \
  --doc_stride=128 \
  --vocab_file=$MODEL_DIR/bert-base-uncased/vocab.txt \
  --config_file=$MODEL_DIR/bert-base-uncased/config.json \
  --json-summary=$MODEL_DIR/dllogger.json \
  --output_dir=$MODEL_DIR/bert-base-uncased-PTQ-mode-2-for-KD \
  --max_steps=-1 \
  --fp16 \
  --calibrator percentile \
  --percentile 99.99 \
  --ft_mode 2

python -m torch.distributed.launch --nproc_per_node=8 run_squad.py \
  --init_checkpoint=$MODEL_DIR/bert-base-uncased-PTQ-mode-2-for-KD/pytorch_model.bin \
  --do_train \
  --train_file=$DATA_DIR/v1.1/train-v1.1.json \
  --train_batch_size=4 \
  --learning_rate=3e-5 \
  --num_train_epochs=10 \
  --do_predict \
  --predict_file=$DATA_DIR/v1.1/dev-v1.1.json \
  --eval_script=$DATA_DIR/v1.1/evaluate-v1.1.py \
  --do_eval \
  --do_lower_case \
  --bert_model=bert-base-uncased \
  --max_seq_length=384 \
  --doc_stride=128 \
  --vocab_file=$MODEL_DIR/bert-base-uncased/vocab.txt \
  --config_file=$MODEL_DIR/bert-base-uncased/config.json \
  --json-summary=$MODEL_DIR/dllogger.json \
  --output_dir=$MODEL_DIR/bert-base-uncased-QAT-mode-2 \
  --max_steps=-1 \
  --fp16 \
  --ft_mode 2 \
  --distillation \
  --teacher=$MODEL_DIR/bert-base-uncased-finetuned/pytorch_model.bin
```

The results would be like:

```bash
{"exact_match": 83.96, "f1": 90.37}
```
