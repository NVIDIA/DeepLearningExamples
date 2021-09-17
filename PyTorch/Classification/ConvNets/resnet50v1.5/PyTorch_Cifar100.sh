# nvidia-docker run --rm -it --ipc=host -v /work/chauhans/cifar100:/cifar100 nvidia_resnet50

for (( i = 0; i < 6; i++ )); do # 5 runs
    mkdir /cifar100/PyTorch/run_"${i}"/
    echo "Created Folder run_${i}"
    python ./multiproc.py --nproc_per_node 2 ./main.py --arch resnet50 --label-smoothing 0.1 --amp --static-loss-scale 256 --data-backend pytorch --num-classes 100 -j 20 --epochs 90 -b 192 --warmup 0 --raport-file /cifar100/PyTorch/run_"${i}"/run_"${i}".json --workspace /cifar100/PyTorch/run_"${i}" /cifar100
    echo "Run ${i} done"
done

