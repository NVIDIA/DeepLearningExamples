mkdir -p output
python -m multiproc train.py -m WaveGlow -o output/ -lr 1e-4 --epochs 1001 -bs 4 --segment-length  8000 --weight-decay 0 --grad-clip-thresh 3.4028234663852886e+38 --cudnn-benchmark --cudnn-enabled --log-file nvlog.json
