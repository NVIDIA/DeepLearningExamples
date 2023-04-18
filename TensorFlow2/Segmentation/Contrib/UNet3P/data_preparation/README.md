For data two options are available

- [Train on LiTS Data](#lits-liver-tumor-segmentation-challenge)
- [Train on custom data](#train-on-custom-data)

## LiTS Liver Tumor Segmentation challenge

This dataset consist of 131 Liver CT Scans.

Register [here](https://competitions.codalab.org/competitions/17094) to get dataset access.
Go to participate &rarr; Training Data to get dataset link.
Download Training Batch 1 and Training Batch 2 zip files and past them under data folder.

`Training Batch 1` size is 3.97GB and `Training Batch 2` zip file size is 11.5GB.

Inside main directory `/workspace/unet3p` run below command to extract zip files

```shell
bash data_preparation/extract_data.sh
```

After extraction `Training Batch 1` folder size will be 11.4GB and `Training Batch 2` folder size will be 38.5GB.

- `Training Batch 1` consist of 28 scans which are used for testing
- `Training Batch 2` consist of 103 scans which are used for training

Default directory structure looks like this

    ├── data/
    │   ├── Training Batch 1/
            ├── segmentation-0.nii
            ├── volume-0.nii
            ├── ...
            ├── volume-27.nii
    │   ├── Training Batch 2/
            ├── segmentation-28.nii
            ├── volume-28.nii
            ├── ...
            ├── volume-130.nii

For testing, you can have any number of files in Training Batch 1 and Training Batch 2. But make sure the naming
convention is similar.

To prepare LiTS dataset for training run

```
python data_preparation/preprocess_data.py
```

> Note: Because of the extensive preprocessing, it will take some time, so relax and wait.

#### Final directory

After completion, you will have a directories like this

    ├── data/
    │   ├── train/
            ├── images
                ├── image_28_0.png
                ├── ...
            ├── mask
                ├── mask_28_0.png
                ├── ...
    │   ├── val/
            ├── images
                ├── image_0_0.png
                ├── ...
            ├── mask
                ├── mask_0_0.png
                ├── ...

After processing the `train` folder size will be 5GB and `val` folder size will be 1.7GB.

#### Free space (Optional)

At this stage you can delete the intermediate scans files to free space, run below command

```shell
bash data_preparation/delete_extracted_scans_data.sh
```

You can also delete the data zip files using below command, but remember you cannot retrieve them back

```shell
bash data_preparation/delete_zip_data.sh
```

> Note: It is recommended to delete scan files but not zip data because you may need it again.

## Train on custom data

To train on custom dateset it's advised that you follow the same train and val directory structure like
mentioned [above](#final-directory).

In our case image file name can be mapped to it's corresponding mask file name by replacing `image` text with `mask`. If
your data has different mapping then you need to update [image_to_mask_name](./../utils/images_utils.py#L63) function which
is responsible for converting image name to it's corresponding file name.

Each image should be a color image with 3 channels and `RGB` color format. Each mask is considered as a gray scale
image, where each pixel value is the class on which each pixel belongs.

Congratulations, now you can start training and testing on your new dataset!