# TensorFlow BERT Quantization Example

Based on [link](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)

Original README: [link](README_orig.md)

Modified the following files:
 * modeling.py
 * run_squad.py

Quantized tensors:
 * encoder
   * linear layer inputs and weights
   * matmul inputs
   * (residual add inputs)
 * final encoder output

Hardware settings:
 * 4 x Tesla V100-SXM2-16GB (with mclk 877MHz, pclk 1530MHz)

## Setup

The docker `nvcr.io/nvidia/tensorflow:20.03-tf1-py3` is used for test (TensorFlow 1.15.2)

setup steps:
```
cd ft-tensorflow-quantization
pip install .
cd ..
export TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE=0
```

## Download pretrained bert checkpoint and SQuAD dataset

Download pretrained bert checkpoint.

```bash
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -O uncased_L-12_H-768_A-12.zip
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
mpirun -np 4 -H localhost:4 \
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
    --train_batch_size=8 \
    --learning_rate=2e-5 \
    --num_train_epochs=2.0 \
    --save_checkpoints_steps 1000 \
    --horovod

python ../sample/tensorflow_bert/squad_evaluate-v1.1.py squad_data/dev-v1.1.json squad_model/finetuned_base/predictions.json
```

The results would be like:

```bash
{"exact_match": 81.22043519394512, "f1": 88.73927073782282}
```

### PTQ by calibrating:

### not quantize residual connection:

```bash
python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/finetuned_base/model.ckpt-5474 \
    --output_dir=squad_model/PTQ_noResidualQuant \
    --do_train=False \
    --do_predict=True \
    --do_calib=True \
    --if_quant=True \
    --train_batch_size=16 \
    --calib_batch=256 \
    --calib_method=mse \
    --quantize_residual=False \
    --quant_another_add_input=False

python ../sample/tensorflow_bert/squad_evaluate-v1.1.py squad_data/dev-v1.1.json squad_model/PTQ_noResidualQuant/predictions.json
```

The results would be like:

```bash
{"exact_match": 80.50141911069063, "f1": 88.20028229553168}
```

### quantize residual connection:


```bash
python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/finetuned_base/model.ckpt-5474 \
    --output_dir=squad_model/PTQ_residualQuant \
    --do_train=False \
    --do_predict=True \
    --do_calib=True \
    --if_quant=True \
    --train_batch_size=16 \
    --calib_batch=256 \
    --calib_method=mse \
    --quantize_residual=True \
    --quant_another_add_input=False

python ../sample/tensorflow_bert/squad_evaluate-v1.1.py squad_data/dev-v1.1.json squad_model/PTQ_residualQuant/predictions.json

```

The results would be like:

```bash
{"exact_match": 79.86754966887418, "f1": 87.6547751188654}
```


## Quantization Aware Fine-tuning

If PTQ does not yield an acceptable result you can finetune with quantization to recover accuracy.
We recommend to calibrate the pretrained model and finetune to avoid overfitting:

### not quantize residual connection:

```bash
python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/bert_model.ckpt \
    --output_dir=squad_model/QAT_calibrated_noResidualQuant \
    --do_train=False \
    --do_calib=True \
    --train_batch_size=8 \
    --calib_batch=256 \
    --calib_method=percentile \
    --percentile=99.99 \
    --quantize_residual=False \
    --quant_another_add_input=False


mpirun -np 4 -H localhost:4 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib \
    python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/QAT_calibrated_noResidualQuant/model.ckpt-calibrated \
    --output_dir=squad_model/QAT_noResidualQuant \
    --do_train=True \
    --do_predict=True \
    --if_quant=True \
    --train_batch_size=8 \
    --learning_rate=1.5e-5 \
    --num_train_epochs=2.0 \
    --save_checkpoints_steps 1000 \
    --quantize_residual=False \
    --quant_another_add_input=False \
    --horovod

python ../sample/tensorflow_bert/squad_evaluate-v1.1.py squad_data/dev-v1.1.json squad_model/QAT_noResidualQuant/predictions.json

```

The results would be like:

```bash
{"exact_match": 81.05960264900662, "f1": 88.53779692514824}
```


### quantize residual connection:
```bash
python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/bert_model.ckpt \
    --output_dir=squad_model/QAT_calibrated_residualQuant \
    --do_train=False \
    --do_calib=True \
    --train_batch_size=8 \
    --calib_batch=256 \
    --calib_method=percentile \
    --percentile=99.99 \
    --quantize_residual=True \
    --quant_another_add_input=False


mpirun -np 4 -H localhost:4 \
    --allow-run-as-root -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO \
    -x LD_LIBRARY_PATH \
    -x PATH -mca pml ob1 -mca btl ^openib \
    python run_squad.py \
    --bert_config_file=squad_model/bert_config.json \
    --vocab_file=squad_model/vocab.txt \
    --train_file=squad_data/train-v1.1.json \
    --predict_file=squad_data/dev-v1.1.json \
    --init_checkpoint=squad_model/QAT_calibrated_residualQuant/model.ckpt-calibrated \
    --output_dir=squad_model/QAT_residualQuant \
    --do_train=True \
    --do_predict=True \
    --if_quant=True \
    --train_batch_size=8 \
    --learning_rate=1.5e-5 \
    --num_train_epochs=2.0 \
    --save_checkpoints_steps 1000 \
    --quantize_residual=True \
    --quant_another_add_input=False \
    --horovod


python ../sample/tensorflow_bert/squad_evaluate-v1.1.py squad_data/dev-v1.1.json squad_model/QAT_residualQuant/predictions.json
```

The results would be like:

```bash
{"exact_match": 80.40681173131505, "f1": 88.03153862226932}
```

## Experience

The results of quantization may differ if different seeds are provided. Thus we ran the commands decribed above several times and got the following results:

| | finetuned <f1, exact_match> | PTQ not quantize residual connection <f1, exact_match> | PTQ quantize residual connection <f1, exact_match> | QAT not quantize residual connection <f1, exact_match> | QAT quantize residual connection <f1, exact_match> | 
|:---------:|:---------------------------:|:------------------------------------------------------:|:--------------------------------------------------:|:------------------------------------------------------:|:--------------------------------------------------:| 
| 0 | <88.73, 81.22> | <88.20, 80.50> | <87.65, 79.86> | <88.53, 81.05> | <88.03, 80.40> |
| 1 | <88.47, 81.00> | <87.82, 79.76> | <87.38, 79.08> | <88.48, 81.19> | <88.16, 80.61> |
| 2 | <88.64, 81.15> | <88.08, 80.24> | <87.48, 79.11> | <88.13, 80.62> | <88.18, 80.85> |
| 3 | <88.56, 81.11> | <88.25, 80.69> | <87.92, 80.14> | <88.34, 80.80> | <88.21, 80.66> |
| 4 | <88.51, 81.20> | <87.69, 80.15> | <87.10, 79.09> | <88.35, 80.85> | <88.27, 80.87> |
| 5 | <88.66, 81.25> | <88.44, 80.70> | <87.79, 79.88> | <88.24, 80.84> | <88.01, 80.66> |
| 6 | <88.51, 81.05> | <88.11, 80.43> | <87.70, 79.59> | <87.82, 80.33> | <88.18, 80.58> |
| 7 | <88.12, 80.46> | <87.52, 79.47> | <87.37, 79.14> | <88.18, 80.58> | <88.04, 80.56> |
| 8 | <88.49, 80.86> | <88.08, 80.31> | <87.63, 79.30> | <88.40, 81.15> | <88.32, 80.89> |


