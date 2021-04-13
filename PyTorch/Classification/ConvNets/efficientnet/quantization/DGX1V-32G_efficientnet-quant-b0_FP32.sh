python ./multiproc.py  \
    --nproc_per_node 8 \
        ./quant_main.py  /imagenet \
            --arch efficientnet-quant-b0 \
            --epochs 10 \
            -j5 -p 500  \
            --data-backend pytorch  \
            --optimizer sgd \
            -b 128 \
            --lr 0.0125 \
            --momentum 0.89 \
            --weight-decay 4.50e-05 \
            --lr-schedule cosine \
            --pretrained-from-file "${1}"