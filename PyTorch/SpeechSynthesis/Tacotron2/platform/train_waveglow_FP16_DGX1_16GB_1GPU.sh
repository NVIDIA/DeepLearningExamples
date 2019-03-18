mkdir -p output
python train.py -m WaveGlow -o output/ --fp16-run -lr 1e-4 --epochs 2001 -bs 8 --segment-length 8000 --weight-decay 0 --grad-clip-thresh 65504.0 --epochs-per-checkpoint 50 --cudnn-benchmark=True --log-file output/nvlog.json
