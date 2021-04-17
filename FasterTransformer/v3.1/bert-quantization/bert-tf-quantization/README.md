# TensorFlow BERT Quantization Example

Based on [link](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)

Original README: [link](README_orig.md)

Modified the following files:
 * modeling.py
 * run_squad.py

Hardware settings:
 * 8 x Tesla V100-SXM2-16GB (with mclk 877MHz, pclk 1530MHz)

## Setup

The docker `nvcr.io/nvidia/tensorflow:20.03-tf1-py3` is used for test (TensorFlow 1.15.2)

setup steps:
```
pip install ft-tensorflow-quantization/
export TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=0
```

## Download pretrained bert checkpoint and SQuAD dataset

Download pretrained bert checkpoint.

```bash
wget https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip -O uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
mv uncased_L-12_H-768_A-12 squad_model
```

Download SQuAD dataset

```bash
mkdir squad_data
wget -P squad_data https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget -P squad_data https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

## Post Training Quantization

### Finetune a high precision model with:

```bash
mpirun -np 8 -H localhost:8 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib \
    python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/bert_model.ckpt \
    --output_dir=squad_model/finetuned_base \
    --do_train=True \
    --do_predict=True \
    --if_quant=False \
    --train_batch_size=4 \
    --learning_rate=1e-5 \
    --num_train_epochs=2.0 \
    --save_checkpoints_steps 1000 \
    --horovod

python ../sample/tensorflow_bert/squad_evaluate_v1_1.py squad_data/dev-v1.1.json squad_model/finetuned_base/predictions.json
```

The results would be like:

```bash
{"exact_match": 82.03, "f1": 89.55}
```

### PTQ by calibrating:

`ft_mode` is unified with int8_mode in FasterTransformer, can be one of `1` or `2`.

```bash
python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/finetuned_base/model.ckpt-5474 \
    --output_dir=squad_model/PTQ_mode_2 \
    --do_train=False \
    --do_predict=True \
    --do_calib=True \
    --if_quant=True \
    --train_batch_size=16 \
    --calib_batch=16 \
    --calib_method=percentile \
    --percentile=99.999 \
    --ft_mode=2

python ../sample/tensorflow_bert/squad_evaluate-v1.1.py squad_data/dev-v1.1.json squad_model/PTQ_mode_2/predictions.json
```

The results would be like:

```bash
{"exact_match": 81.68, "f1": 88.97}     # for mode 1
{"exact_match": 80.65, "f1": 88.31}     # for mode 2
```


## Quantization Aware Fine-tuning

If PTQ does not yield an acceptable result you can finetune with quantization to recover accuracy.
We recommend to calibrate the pretrained model and finetune to avoid overfitting:

`ft_mode` is unified with int8_mode in FasterTransformer, can be one of `1` or `2`.

```bash
python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/bert_model.ckpt \
    --output_dir=squad_model/QAT_calibrated_mode_2 \
    --do_train=False \
    --do_calib=True \
    --train_batch_size=16 \
    --calib_batch=16 \
    --calib_method=percentile \
    --percentile=99.99 \
    --ft_mode=2


mpirun -np 8 -H localhost:8 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib \
    python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/QAT_calibrated_mode_2/model.ckpt-calibrated \
    --output_dir=squad_model/QAT_mode_2 \
    --do_train=True \
    --do_predict=True \
    --if_quant=True \
    --train_batch_size=4 \
    --learning_rate=5e-6 \
    --num_train_epochs=2.0 \
    --save_checkpoints_steps 1000 \
    --ft_mode=2
    --horovod

python ../sample/tensorflow_bert/squad_evaluate-v1.1.py squad_data/dev-v1.1.json squad_model/QAT_mode_2/predictions.json
```

The results would be like:

```bash
{"exact_match": 82.17, "f1": 89.34}     # for mode 1
{"exact_match": 81.99, "f1": 89.14}     # for mode 2
```


The results of quantization may differ if different seeds are provided.


## Quantization Aware Fine-tuning with Knowledge-distillation

Knowledge-distillation can get better results, we usually starts from a PTQ checkpoint.

```bash
python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/finetuned_base/model.ckpt-5474 \
    --output_dir=squad_model/PTQ_mode_2_for_KD \
    --do_train=False \
    --do_predict=False \
    --do_calib=True \
    --if_quant=True \
    --train_batch_size=16 \
    --calib_batch=16 \
    --calib_method=percentile \
    --percentile=99.99 \
    --ft_mode=2

mpirun -np 8 -H localhost:8 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib \
    python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/PTQ_mode_2_for_KD/model.ckpt-calibrated \
    --output_dir=squad_model/QAT_KD_mode_2 \
    --do_train=True \
    --do_predict=True \
    --if_quant=True \
    --train_batch_size=4 \
    --learning_rate=5e-6 \
    --num_train_epochs=10.0 \
    --save_checkpoints_steps 1000 \
    --ft_mode=2
    --horovod
    --distillation=True \
    --teacher=squad_model/finetuned_base/model.ckpt-5474

python ../sample/tensorflow_bert/squad_evaluate-v1.1.py squad_data/dev-v1.1.json squad_model/QAT_KD_mode_2/predictions.json
```

The results would be like:

```bash
{"exact_match": 83.56, "f1": 90.22}
```
