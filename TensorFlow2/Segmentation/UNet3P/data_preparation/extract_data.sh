# extract testing data
unzip data/Training_Batch1.zip -d data/
mv  "data/media/nas/01_Datasets/CT/LITS/Training Batch 1/" "data/Training Batch 1/"
rm -r data/media

# extract training data
unzip data/Training_Batch2.zip -d data/
mv  "data/media/nas/01_Datasets/CT/LITS/Training Batch 2/" "data/Training Batch 2/"
rm -r data/media
