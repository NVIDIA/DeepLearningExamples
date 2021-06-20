FLAGS=$1
STAGE_ID=$2
STAGE_LEN=$3


python ./multiproc.py  \
    --nproc_per_node 8 \
        ./main.py /imagenet  \
            -j5 -p 100  \
            --data-backend pytorch  \
            --raport-file report_$STAGE_ID.json \
            --lr 1.024  \
            --batch-size 128 \
            --optimizer-batch-size 1024  \
            --static-loss-scale 128 \
            --warmup 8  \
            --arch resnext101-32x4d -c fanin  \
            --label-smoothing 0.1  \
            --lr-schedule cosine  \
            --mom 0.875  \
            --wd  6.103515625e-05  \
            --workspace /results \
            --epochs 90 \
            --run-epochs $STAGE_LEN \
            $FLAGS \
            --resume /results/checkpoint_$( expr $STAGE_ID - 1).pth.tar \
            --checkpoint checkpoint_$STAGE_ID.pth.tar

