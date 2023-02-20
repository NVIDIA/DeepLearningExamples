Here we provide **overview** of our config file and how you can use your own custom settings's for training and
evaluation.

We are using [Hydra](https://hydra.cc/) for passing configurations. Hydra is a framework for elegantly configuring
complex applications. In Hydra you can easily [extend](https://hydra.cc/docs/patterns/extending_configs/)
and [interpolate](https://hydra.cc/docs/advanced/override_grammar/basic/#primitives) `yaml` config files.

#### Override Hydra config from command line

[Here](https://hydra.cc/docs/1.0/advanced/override_grammar/basic/)  you can read how to pass or override configurations
through command line. Overall to

###### Override higher level attribute

Directly access the key and override its value

- For instance to override Data generator pass `DATA_GENERATOR_TYPE=DALI_GENERATOR`

###### Override nested attribute

Use `.` to access nested keys

- For instance to override model type `MODEL.TYPE=unet3plus`
- To override model backbone `MODEL.BACKBONE.TYPE=vgg19`

To add new element from command line add `+` before attribute name. E.g. `+warmup_steps=50` because warm steps is not
added in config file.

> Note: Don't add space between list elements, it will create problem with Hydra.

Most of the configurations attributes in our [config](./../configs/config.yaml) are self-explanatory. However, for some
attributes additional comments are added.

You can override configurations from command line too, but it's **advisable to override them from config file** because
it's
easy.

By default, hydra stores a log file of each run in a separate directory. We have disabled it in our case,
if you want to enable them to keep record of each run configuration's then comment out the settings at the end of config
file.

```yaml
# project root working directory, automatically read by hydra (.../UNet3P)
WORK_DIR: ${hydra:runtime.cwd}
DATA_PREPARATION:
  # unprocessed LiTS scan data paths, for custom data training skip this section details 
  SCANS_TRAIN_DATA_PATH: "/data/Training Batch 2/"
  ...
DATASET:
  # training data paths, should be relative from project root path
  TRAIN:
    IMAGES_PATH: "/data/train/images"
  ...
MODEL:
  # available variants are unet3plus, unet3plus_deepsup, unet3plus_deepsup_cgm
  TYPE: "unet3plus"
  BACKBONE:
  ...
...
DATA_GENERATOR_TYPE: "DALI_GENERATOR"  # options are TF_GENERATOR or DALI_GENERATOR
SHOW_CENTER_CHANNEL_IMAGE: True  # only true for UNet3+. for custom dataset it should be False
# Model input shape
INPUT:
  HEIGHT: 320
  ...
# Model output classes
OUTPUT:
  CLASSES: 2
HYPER_PARAMETERS:
  EPOCHS: 5
  BATCH_SIZE: 2  # specify per gpu batch size
  ...
CALLBACKS:
  TENSORBOARD:
  ...
PREPROCESS_DATA:
  RESIZE:
    VALUE: False  # if True, resize to input height and width
    ...
USE_MULTI_GPUS:
  ...
# to stop hydra from storing logs files
defaults:
  ...

```
