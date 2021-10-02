# nvidia-docker run --rm -it --ipc=host -v /work/chauhans/cifar100:/cifar100 mxnet_cifar100
for (( i = 0; i < 6; i++ )); do # 5 runs
    mkdir /cifar100/MxNet/run_"${i}"/
    echo "Created Folder run_${i}"
    ./runner -n 2 -b 192 --num-epochs 90 --num-classes 100 --mode train_val --amp --dllogger-log /cifar100/MxNet/run_"${i}"/run_"${i}".log --workspace /cifar100/MxNet/run_"${i}"/ --data-backend mxnet --data-root /cifar100 --warmup 0
    echo "Run ${i} done"
done
