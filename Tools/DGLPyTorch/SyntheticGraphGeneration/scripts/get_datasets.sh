#Note: Each user is responsible for checking the content of datasets and the applicable licenses and determining if suitable for the intended use

if [ ! "$(ls | grep -c ^scripts$)" -eq 1 ]; then
  echo "Run this script from root directory. Usage: bash ./scripts/get_datasets.sh"
  exit 1
fi

mkdir -p data
cd data || exit 1

# Lastfm
echo "Processing lastfm ..."
echo "@inproceedings{feather,
title={{Characteristic Functions on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric Models}},
author={Benedek Rozemberczki and Rik Sarkar},
year={2020},
pages = {1325–1334},
booktitle={Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM '20)},
organization={ACM},
}"
if [ "$(ls | grep -c "^lasftm_asia$")" -ge 1 ]; then
  echo "Lastfm directory already exists, skipping ..."
else
  wget https://snap.stanford.edu/data/lastfm_asia.zip
  unzip lastfm_asia.zip
  rm lastfm_asia.zip
fi


# Twitch
echo "Processing Twitch ..."
echo "@misc{rozemberczki2019multiscale,
  title={Multi-scale Attributed Node Embedding},
  author={Benedek Rozemberczki and Carl Allen and Rik Sarkar},
  year={2019},
  eprint={1909.13021},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}"
if [ "$(ls | grep -c "^twitch$")" -ge 1 ]; then
  echo "Twitch directory already exists, skipping ..."
else
  mkdir -p twitch && cd twitch || exit 1
  wget https://snap.stanford.edu/data/twitch_gamers.zip && unzip twitch_gamers.zip
  rm twitch_gamers.zip
  cd ..
fi


# Orkut
echo "Processing Orkut ..."
echo "@inproceedings{yang2012defining,
  title={Defining and evaluating network communities based on ground-truth},
  author={Yang, Jaewon and Leskovec, Jure},
  booktitle={Proceedings of the ACM SIGKDD Workshop on Mining Data Semantics},
  pages={1--8},
  year={2012}
}"
if [ "$(ls | grep -c "^orkut$")" -ge 1 ]; then
  echo "Orkut directory already exists, skipping ..."
else
  mkdir -p orkut && cd orkut || exit 1
  wget https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz && gzip -d com-orkut.ungraph.txt.gz
  rm com-orkut.ungraph.txt.gz
  cd ..
fi


# Tabformer
echo "Processing tabformer ..."
echo "@inproceedings{padhi2021tabular,
  title={Tabular transformers for modeling multivariate time series},
  author={Padhi, Inkit and Schiff, Yair and Melnyk, Igor and Rigotti, Mattia and Mroueh, Youssef and Dognin, Pierre and Ross, Jerret and Nair, Ravi and Altman, Erik},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={3565--3569},
  year={2021},
  organization={IEEE},
  url={https://ieeexplore.ieee.org/document/9414142}
}"
if [ "$(ls | grep -c "^tabformer$")" -ge 1 ]; then
  echo "Tabformer directory already exists, skipping ..."
else
  if [ "$(ls | grep -c "^transactions.tgz$")" -eq 0 ]; then
    echo "transactions.tgz not found, skipping ..."
    echo "Download tabformer manually - https://github.com/IBM/TabFormer/tree/main/data/credit_card/ and store it as ./data/transactions.tgz"
  else
    mkdir -p tabformer && mv transactions.tgz tabformer && cd tabformer || exit 1
    tar zxvf transactions.tgz
    mv transactions.tgz ..
    python ../../scripts/time_filter_tabformer.py ./card_transaction.v1.csv
    rm card_transaction.v1.csv
    cd ..
  fi
fi


# IEEE
echo "Processing IEEE ..."
# kaggle competitions download -c ieee-fraud-detection
if [ "$(ls | grep -c "^ieee-fraud$")" -ge 1 ]; then
  echo "IEEE directory already exists, skipping ..."
else
  if [ "$(ls | grep -c "^ieee-fraud-detection.zip$")" -eq 0 ]; then
    echo "ieee-fraud-detection.zip not found, skipping ..."
    echo "Download IEEE manually from https://www.kaggle.com/competitions/ieee-fraud-detection/data and store it as ./data/ieee-fraud-detection.zip"
    # kaggle competitions download -c ieee-fraud-detection // exemplary command to download
  else
    mkdir -p ieee-fraud && mv ieee-fraud-detection.zip ieee-fraud && cd ieee-fraud || exit 1
    unzip ieee-fraud-detection.zip "*_transaction.csv"
    mv ieee-fraud-detection.zip ..
    python ../../scripts/ieee_fraud.py .
    rm *_transaction.csv
    cd ..
  fi
fi



# Paysim
echo "Processing Paysim ..."
if [ "$(ls |  grep -c "^paysim$")" -ge 1 ]; then
  echo "Paysim directory already exists, skipping ..."
else
  if [ "$(ls | grep -c "^paysim.zip$")" -eq 0 ]; then
    echo "paysim.zip not found, skipping ..."
    echo "Download paysim manually from https://www.kaggle.com/datasets/ealaxi/paysim1/download?datasetVersionNumber=2 and store it as ./data/paysim.zip"
    #kaggle datasets download -d ealaxi/paysim1 #exemplary command to download
  else
    mkdir -p paysim && mv paysim.zip paysim && cd paysim || exit 1
    unzip paysim.zip
    mv paysim.zip ..
    cd ..
  fi
fi



# credit
echo "Processing credit ..."
if [ "$(ls | grep "^credit$")" -ge 1 ]; then
  echo "credit directory already exists, skipping ..."
else
  if [ "$(ls | grep -c "^credit.zip$")" -eq 0 ]; then
    echo "credit.zip not found, skipping ..."
    echo "Download credit manually from https://www.kaggle.com/datasets/kartik2112/fraud-detection/download?datasetVersionNumber=1 and store it as ./data/credit.zip"
    # kaggle datasets download -d kartik2112/fraud-detection // exemplary command to download
  else
    mkdir -p credit && mv credit.zip credit && cd credit || exit 1
    unzip credit.zip "fraudTrain.csv"
    mv credit.zip ..
    python ../../scripts/time_filter_credit.py ./fraudTrain.csv
    rm "fraudTrain.csv"
    cd ..
  fi
fi



# CORA
echo "Processing CORA ..."
echo "@article{sen:aim08,
  title = {Collective Classification in Network Data},
  author = {Prithviraj Sen, Galileo Mark Namata, Mustafa Bilgic, Lise Getoor, Brian Gallagher, and Tina Eliassi-Rad},
  journal = {AI Magazine},
  year = {2008},
  publisher = {AAAI},
  pages = {93--106},
  volume = {29},
  number = {3},
}"
if [ "$(ls | grep -c "^cora$")" -ge 1 ]; then
  echo "CORA directory already exists, skipping ..."
else
  python -m syngen preprocess --source-path=./cora --dataset=cora --download
fi


# Rating
echo "Processing Rating ..."

if [ "$(ls | grep -c "^epinions$")" -ge 1 ]; then
  echo "Rating file already exists, skipping ..."
else
  python -m syngen preprocess --source-path=./epinions --dataset=epinions --download
fi
