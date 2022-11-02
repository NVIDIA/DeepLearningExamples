DATA_DIR=$1

wget -c http://images.cocodataset.org/zips/train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

if md5sum -c hashes.md5
then
	echo "DOWNLOAD PASSED"
	# mkdir $DATA_DIR
	mv train2017.zip $DATA_DIR
	mv val2017.zip $DATA_DIR
	mv annotations_trainval2017.zip $DATA_DIR

	cd $DATA_DIR
	dtrx --one=here annotations_trainval2017.zip
	dtrx train2017.zip
	dtrx val2017.zip

	echo "EXTRACTION COMPLETE"
else
	echo "DOWNLOAD FAILED HASHCHECK"
fi
