# nvidia-docker run --rm -it --ipc=host -v /work/chauhans/imagenet:/imagenet mxnet_cifar100
for (( i = 0; i < 3; i++ )); do # 3 runs
    mkdir /imagenet/MxNet/run_"${i}"/
    echo "Created Folder run_${i}"
    ./runner -n 2 -b 192 --num-epochs 90 --mode train_val \
    --amp --dllogger-log /imagenet/MxNet/run_"${i}"/run_"${i}".log \
    --workspace /imagenet/MxNet/run_"${i}"/ --data-backend mxnet \
    --data-root /imagenet --warmup 0
    echo "Run ${i} done"
done
