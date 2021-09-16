# nvidia-docker run --rm -it --ipc=host -v /work/chauhans/cifar100:/cifar100 nvidia_rn50_mx
./runner -n 2 -b 192 --num-epochs 90 --num-classes 100 --mode train_val --amp --dllogger-log /cifar100/MxNet/MX_Cifar100.log --workspace /cifar100/MxNet --data-backend synthetic --image-shape 3,32,32 --data-root /cifar100/
