

./runner -n 2 -b 192 --num-epochs 90 --num-classes 10 --mode train_val --amp --dllogger-log /cifar10/MxNet/MX_Cifar10.log --workspace /cifar10/MxNet --data-backend mxnet --image-shape 3,32,32 --data-root /cifar10 --warmup 0