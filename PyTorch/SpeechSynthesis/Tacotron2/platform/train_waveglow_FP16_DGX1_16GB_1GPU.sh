mkdir -p output
python train.py -m WaveGlow -o output/ --fp16-run -lr 1e-4 --epochs 2001 -bs 10 --segment-length 8000 --weight-decay 0 --grad-clip-thresh 65504.0 --cudnn-benchmark --cudnn-enabled --log-file output/nvlog.json
