FLAGS=$1
STAGE_ID=$2
STAGE_LEN=$3


python ./multiproc.py  \
    --nproc_per_node 8 \
        ./main.py /imagenet  \
            -j5 -p 100  \
            --data-backend pytorch  \
            --raport-file report_$STAGE_ID.json \
            --lr 2.048  \
            --batch-size 256 \
            --optimizer-batch-size 2048  \
            --static-loss-scale 128 \
            --warmup 8  \
            --arch resnet50 -c fanin  \
            --label-smoothing 0.1  \
            --lr-schedule cosine  \
            --mom 0.875  \
            --wd 3.0517578125e-05  \
            --workspace /results \
            --epochs 90 \
            --run-epochs $STAGE_LEN \
            $FLAGS \
            --resume /results/checkpoint_$( expr $STAGE_ID - 1).pth.tar \
            --checkpoint checkpoint_$STAGE_ID.pth.tar

