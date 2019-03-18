mkdir -p output
python train.py -m Tacotron2 -o output/ --fp16-run -lr 1e-3 --epochs 2001 -bs 80 --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-benchmark=True --log-file output/nvlog.json --anneal-steps 500 1000 1500 --anneal-factor 0.3
