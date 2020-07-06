mkdir -p output
python -m multiproc train.py -m WaveGlow -o output/ --amp -lr 1e-4 --epochs 1001 -bs 10 --segment-length 8000 --weight-decay 0 --grad-clip-thresh 65504.0 --cudnn-benchmark --cudnn-enabled --log-file nvlog.json
