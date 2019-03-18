DATASET_NAME=$1
RAW_DATADIR=$2

function download_20m {
	echo "Download ml-20m"
	curl -O http://files.grouplens.org/datasets/movielens/ml-20m.zip
	mv ml-20m.zip ${RAW_DATADIR}
}

function download_1m {
	echo "Downloading ml-1m"
	curl -O http://files.grouplens.org/datasets/movielens/ml-1m.zip
	mv ml-1m.zip ${RAW_DATADIR}
}

if [[ ${DATASET_NAME} == "ml-1m" ]]
then
	download_1m
elif [[ ${DATASET_NAME} == "ml-20m" ]]
then
    download_20m
else
	echo "Unsupported dataset name: $DATASET_NAME"
	exit 1
fi
