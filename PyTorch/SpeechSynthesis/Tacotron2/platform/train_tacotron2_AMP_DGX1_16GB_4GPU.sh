mkdir -p output
python -m multiproc train.py -m Tacotron2 -o output/ --amp-run -lr 1e-3 --epochs 1501 -bs 104 --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --load-mel-from-disk --training-files=filelists/ljs_mel_text_train_filelist.txt --validation-files=filelists/ljs_mel_text_val_filelist.txt --log-file nvlog.json --anneal-steps 500 1000 1500 --anneal-factor 0.3
