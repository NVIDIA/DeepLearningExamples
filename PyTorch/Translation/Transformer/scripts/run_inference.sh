: ${FP16:=0}

[ ${FP16} -ne 0 ] && PREC="--fp16"

sacrebleu -t wmt14/full -l en-de --echo src | \
python inference.py \
    --buffer-size 5000 \
    --path /checkpoints/transformer_pyt_20.06.pt \
    --max-tokens 10240 \
    --fuse-dropout-add \
    --remove-bpe \
    --bpe-codes /checkpoints/bpe_codes \
    ${PREC} \
    | sacrebleu -t wmt14/full -l en-de -lc

