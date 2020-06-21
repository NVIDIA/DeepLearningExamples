# This script does Quantization aware training of Resnet-50 by finetuning on the pre-trained model using 1 GPU and a batch size of 32.
# Usage ./GPU1_RN50_QAT.sh <path to the pre-trained model> <path to dataset> <path to results directory>

python main.py --mode=train_and_evaluate --batch_size=32 --lr_warmup_epochs=1 --label_smoothing 0.1 --lr_init=0.00005 --momentum=0.875 --weight_decay=3.0517578125e-05 --finetune_checkpoint=$1 --data_dir=$2 --results_dir=$3 --quantize --symmetric --num_iter 10 --data_format NHWC
