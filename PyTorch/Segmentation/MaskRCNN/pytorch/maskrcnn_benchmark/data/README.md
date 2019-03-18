# Setting Up Datasets
This file describes how to perform training on other datasets.

Only Pascal VOC dataset can be loaded from its original format and be outputted to Pascal style results currently.

We expect the annotations from other datasets be converted to COCO json format, and
the output will be in COCO-style. (i.e. AP, AP50, AP75, APs, APm, APl for bbox and segm)

## Creating Symlinks for PASCAL VOC

We assume that your symlinked `datasets/voc/VOC<year>` directory has the following structure:

```
VOC<year>
|_ JPEGImages
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ Annotations
|  |_ pascal_train<year>.json (optional)
|  |_ pascal_val<year>.json (optional)
|  |_ pascal_test<year>.json (optional)
|  |_ <im-1-name>.xml
|  |_ ...
|  |_ <im-N-name>.xml
|_ VOCdevkit<year>
```

Create symlinks for `voc/VOC<year>`:

```
cd ~/github/maskrcnn-benchmark
mkdir -p datasets/voc/VOC<year>
ln -s /path/to/VOC<year> /datasets/voc/VOC<year>
```
Example configuration files for PASCAL VOC could be found [here](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/configs/pascal_voc/).

### PASCAL VOC Annotations in COCO Format
To output COCO-style evaluation result, PASCAL VOC annotations in COCO json format is required and could be downloaded from [here](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip)
via http://cocodataset.org/#external.

## Creating Symlinks for Cityscapes:

We assume that your symlinked `datasets/cityscapes` directory has the following structure:

```
cityscapes
|_ images
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ annotations
|  |_ instanceonly_gtFile_train.json
|  |_ ...
|_ raw
   |_ gtFine
   |_ ...
   |_ README.md
```

Create symlinks for `cityscapes`:

```
cd ~/github/maskrcnn-benchmark
mkdir -p datasets/cityscapes
ln -s /path/to/cityscapes datasets/data/cityscapes
```

### Steps to convert Cityscapes Annotations to COCO Format
1. Download gtFine_trainvaltest.zip from https://www.cityscapes-dataset.com/downloads/ (login required)
2. Extract it to /path/to/gtFine_trainvaltest
```
gtFine_trainvaltest
|_ gtFine
```
3. Run the below commands to convert the annotations

```
cd ~/github
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts
cp ~/github/maskrcnn-benchmark/tool/cityscapes/instances2dict_with_polygons.py cityscapesscripts/evaluation
python setup.py install
cd ~/github/maskrcnn-benchmark
python tools/cityscapes/convert_cityscapes_to_coco.py --datadir /path/to/gtFine_trainvaltest --outdir /path/to/cityscapes/annotations
```

Example configuration files for Cityscapes could be found [here](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/configs/cityscapes/).
