python /opt/mxnet/tools/im2rec.py --list --recursive train /cifar100/train
python /opt/mxnet/tools/im2rec.py --list --recursive val /cifar100/val
python /opt/mxnet/tools/im2rec.py --pass-through --num-thread 40 train /cifar100/train
python /opt/mxnet/tools/im2rec.py --pass-through --num-thread 40 val /cifar100/val
