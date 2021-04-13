python ./multiproc.py  \
    --nproc_per_node 8 \
        ./quant_main.py  /imagenet \
            --arch efficientnet-quant-b4 \
            --epochs 2 \
            -j5 -p 500  \
            --data-backend pytorch  \
            --optimizer rmsprop \
            -b 32 \
            --lr 4.09e-06 \
            --momentum 0.9 \
            --weight-decay 9.714e-04 \
            --lr-schedule linear \
            --rmsprop-alpha 0.853 \
            --rmsprop-eps 0.00422 \
            --pretrained-from-file "${1}"
