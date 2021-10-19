# nvidia-docker run --rm -it --ipc=host -v /work/chauhans/imagenet:/imagenet nvidia_resnet50

for (( i = 0; i < 3; i++ )); do # 3 runs
    mkdir /imagenet/PyTorch/run_"${i}"/
    echo "Created Folder run_${i}"
    python ./multiproc.py --nproc_per_node 2 ./main.py --arch resnet50 \
    --amp --data-backend pytorch -j 20 --epochs 90 -b 192 \
    --warmup 0 --raport-file /imagenet/PyTorch/run_"${i}"/run_"${i}".json \
    --workspace /imagenet/PyTorch/run_"${i}" /imagenet \
    --seed "${i}"
    echo "Run ${i} done"
done
