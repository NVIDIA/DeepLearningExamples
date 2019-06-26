mkdir -p output
python -m multiproc train.py -m WaveGlow -o ./output/ -lr 1e-4 --epochs 1000 -bs 8 --segment-length  8000 --weight-decay 0 --grad-clip-thresh 65504.0 --epochs-per-checkpoint 50 --cudnn-benchmark=True --log-file ./output/nvlog.json --fp16-run
