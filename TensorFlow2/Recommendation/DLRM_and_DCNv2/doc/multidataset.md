# BYO dataset functionality overview

This section describes how you can train the DeepLearningExamples RecSys models on your own datasets without changing
the model or data loader and with similar performance to the one published in each repository.
This can be achieved thanks to Dataset Feature Specification, which describes how the dataset, data loader and model
interact with each other during training, inference and evaluation.
Dataset Feature Specification has a consistent format across all recommendation models in NVIDIA’s DeepLearningExamples
repository, regardless of dataset file type and the data loader,
giving you the flexibility to train RecSys models on your own datasets.

- [Glossary](#glossary)
- [Dataset Feature Specification](#dataset-feature-specification)
- [Data Flow in Recommendation Models in DeepLearning examples](#data-flow-in-nvidia-deep-learning-examples-recommendation-models)
- [Example of Dataset Feature Specification](#example-of-dataset-feature-specification)
- [BYO dataset functionality](#byo-dataset-functionality)

## Glossary

The Dataset Feature Specification consists of three mandatory and one optional section:

<b>feature_spec </b> provides a base of features that may be referenced in other sections, along with their metadata.
	Format: dictionary (feature name) => (metadata name => metadata value)<br>

<b>source_spec </b> provides information necessary to extract features from the files that store them. 
	Format: dictionary (mapping name) => (list of chunks)<br>

* <i>Mappings</i> are used to represent different versions of the dataset (think: train/validation/test, k-fold splits). A mapping is a list of chunks.<br>
* <i>Chunks</i> are subsets of features that are grouped together for saving. For example, some formats may constrain data saved in one file to a single data type. In that case, each data type would correspond to at least one chunk. Another example where this might be used is to reduce file size and enable more parallel loading. Chunk description is a dictionary of three keys:<br>
  * <i>type</i> provides information about the format in which the data is stored. Not all formats are supported by all models.<br>
  * <i>features</i> is a list of features that are saved in a given chunk. Order of this list may matter: for some formats, it is crucial for assigning read data to the proper feature.<br>
  * <i>files</i> is a list of paths to files where the data is saved. For Feature Specification in yaml format, these paths are assumed to be relative to the yaml file’s directory (basename). <u>Order of this list matters:</u> It is assumed that rows 1 to i appear in the first file, rows i+1 to j in the next one, etc. <br>

<b>channel_spec</b> determines how features are used. It is a mapping (channel name) => (list of feature names). 

Channels are model specific magic constants. In general, data within a channel is processed using the same logic. Example channels: model output (labels), categorical ids, numerical inputs, user data, and item data.

<b>metadata</b> is a catch-all, wildcard section: If there is some information about the saved dataset that does not fit into the other sections, you can store it here.

## Dataset feature specification

Data flow can be described abstractly:
Input data consists of a list of rows. Each row has the same number of columns; each column represents a feature.
The columns are retrieved from the input files, loaded, aggregated into channels and supplied to the model/training script. 

FeatureSpec contains metadata to configure this process and can be divided into three parts:

* Specification of how data is organized on disk (source_spec). It describes which feature (from feature_spec) is stored in which file and how files are organized on disk.

* Specification of features (feature_spec). Describes a dictionary of features, where key is feature name and values are features’ characteristics such as  dtype and other metadata (for example, cardinalities for categorical features)

* Specification of model’s inputs and outputs (channel_spec). Describes a dictionary of model’s inputs where keys specify model channel’s names and values specify lists of features to be loaded into that channel. Model’s channels are groups of data streams to which common model logic is applied, for example categorical/continuous data, user/item ids. Required/available channels depend on the model


The FeatureSpec is a common form of description regardless of underlying dataset format, dataset data loader form and model. 


## Data flow in NVIDIA Deep Learning Examples recommendation models

The typical data flow is as follows:
* <b>S.0.</b> Original dataset is downloaded to a specific folder.
* <b>S.1.</b> Original dataset is preprocessed into Intermediary Format. For each model, the preprocessing is done differently, using different tools. The Intermediary Format also varies (for example, for DLRM PyTorch, the Intermediary Format is a custom binary one.)
* <b>S.2.</b> The Preprocessing Step outputs Intermediary Format with dataset split into training and validation/testing parts along with the Dataset Feature Specification yaml file. Metadata in the preprocessing step is automatically calculated.
* <b>S.3.</b> Intermediary Format data together with Dataset Feature Specification are fed into training/evaluation scripts. Data loader reads Intermediary Format and feeds the data into the model according to the description in the Dataset Feature Specification.
* <b>S.4.</b> The model is trained and evaluated



<p align="center">
  <img width="70%" src="./img/df_diagram.png" />
  <br>

Fig.1. Data flow in Recommender models in NVIDIA Deep Learning Examples repository. Channels of the model are drawn in green</a>.
</p>


### Example of dataset feature specification

As an example, let’s consider a Dataset Feature Specification for a small CSV dataset for some abstract model.

```yaml
feature_spec:
  user_gender:
    dtype: torch.int8
    cardinality: 3 #M,F,Other
  user_age: #treated as numeric value
    dtype: torch.int8
  user_id:
    dtype: torch.int32
    cardinality: 2655
  item_id:
    dtype: torch.int32
    cardinality: 856
  label:
    dtype: torch.float32

source_spec:
  train:
    - type: csv
      features:
        - user_gender
        - user_age
      files:
        - train_data_0_0.csv
        - train_data_0_1.csv
    - type: csv
      features:
        - user_id
        - item_id
        - label
      files:
        - train_data_1.csv
  test:
    - type: csv
      features:
        - user_id
        - item_id
        - label
        - user_gender
        - user_age
        
      files:
        - test_data.csv

channel_spec:
  numeric_inputs: 
    - user_age
  categorical_user_inputs: 
    - user_gender
    - user_id
  categorical_item_inputs: 
    - item_id
  label_ch: 
    - label
```


The data contains five features: (user_gender, user_age, user_id, item_id, label). Their data types and necessary metadata are described in the feature specification section.

In the source mapping section, two mappings are provided: one describes the layout of the training data, the other of the testing data. The layout for training data has been chosen arbitrarily to showcase the flexibility.
The train mapping consists of two chunks. The first one contains user_gender and user_age, saved as a CSV, and is further broken down into two files. For specifics of the layout, refer to the following example and consult the glossary. The second chunk contains the remaining columns and is saved in a single file. Notice that the order of columns is different in the second chunk - this is alright, as long as the order matches the order in that file (that is, columns in the .csv are also switched)


Let’s break down the train source mapping. The table contains example data color-paired to the files containing it.

<p align="center">
<img width="70%" src="./img/layout_example.png" />
</p>



The channel spec describes how the data will be consumed. Four streams will be produced and available to the script/model.
The feature specification does not specify what happens further: names of these streams are only lookup constants defined by the model/script.
Based on this example, we can speculate that the model has three  input channels: numeric_inputs, categorical_user_inputs,
categorical_item_inputs, and one  output channel: label.
Feature names are internal to the FeatureSpec and can be freely modified.


### BYO dataset functionality

In order to train any Recommendation model in NVIDIA Deep Learning Examples one can follow one of three possible ways:
* One delivers already preprocessed dataset in the Intermediary Format supported by data loader used by the training script
(different models use different data loaders) together with FeatureSpec yaml file describing at least specification of dataset, features and model channels

* One uses a transcoding script

* One delivers dataset in non-preprocessed form and uses preprocessing scripts that are a part of the model repository.
In order to use already existing preprocessing scripts, the format of the dataset needs to match the one of the original datasets.
This way, the FeatureSpec file will be generated automatically, but the user will have the same preprocessing as in the original model repository.



### BYO dataset 

The BYO dataset functionality allows users to plug in their dataset in a common fashion for all Recommender models 
that support this functionality. Using BYO dataset functionality, the user does not have to modify the source code of 
the model thanks to the Feature Specification file. For general information on how BYO dataset works, refer to the 
[BYO dataset overview section](#byo-dataset-functionality-overview).

There are three ways to plug in user's dataset:
<details>
<summary><b>1. Provide an unprocessed dataset in a format matching the one used by Criteo 1TB, then use Criteo 1TB's preprocessing. Feature Specification file is then generated automatically.</b></summary>
The required format of the user's dataset is:

The data should be split into text files. Each line of those text files should contain a single training example. 
An example should consist of multiple fields separated by tabulators:

* The first field is the label – 1 for a positive example and 0 for negative.
* The next N tokens should contain the numerical features separated by tabs.
* The next M tokens should contain the hashed categorical features separated by tabs.

The correct dataset files together with the Feature Specification yaml file will be generated automatically by preprocessing script.

For an example of using this process, refer to the [Quick Start Guide](#quick-start-guide)

</details>

<details>
<summary><b>2. Provide a CSV containing preprocessed data and a simplified Feature Specification yaml file, then transcode the data with `transcode.py` script </b> </summary>
This option should be used if the user has their own CSV file with a preprocessed dataset they want to train on.

The required format of the user's dataset is:
* CSV files containing the data, already split into train and test sets. 
* Feature Specification yaml file describing the layout of the CSV data

For an example of a feature specification file, refer to the `tests/transcoding` folder.

The CSV containing the data:
* should be already split into train and test
* should contain no header
* should contain one column per feature, in the order specified by the list of features for that chunk 
  in the source_spec section of the feature specification file
* categorical features should be non-negative integers in the range [0,cardinality-1] if cardinality is specified

The Feature Specification yaml file:
* needs to describe the layout of data in CSV files
* may contain information about cardinalities. However, if set to `auto`, they will be inferred from the data by the transcoding script.

Refer to `tests/transcoding/small_csv.yaml` for an example of the yaml Feature Specification.

The following example shows how to use this way of plugging user's dataset:

Prepare your data and save the path:
```bash
DATASET_PARENT_DIRECTORY=/raid/dlrm
```

Build the DLRM image with:
```bash
docker build -t nvidia_dlrm_tf .
```
Launch the container with:
```bash
docker run --cap-add SYS_NICE --runtime=nvidia -it --rm --ipc=host  -v ${DATASET_PARENT_DIRECTORY}/data:/data nvidia_dlrm_tf bash
```

If you are just testing the process, you can create synthetic csv data:
```bash
python gen_csv.py --feature_spec_in tests/transcoding/small_csv.yaml
```

Convert the data:
```bash
mkdir /data/conversion_output
cp tests/transcoding/small_csv.yaml /data/feature_spec.yaml
python transcode.py --input /data --output /data/converted
```
You may need to tune the --chunk_size parameter. Higher values speed up the conversion but require more RAM.

This will convert the data from `/data` and save the output in `/data/converted`.
A feature specification file describing the new data will be automatically generated.

To run the training on 1 GPU:
```bash
horovodrun -np 1 -H localhost:1 --mpi-args=--oversubscribe numactl --interleave=all -- python -u main.py --dataset_path /data/converted --amp --xla
```

- multi-GPU for DGX A100:
```bash
horovodrun -np 8 -H localhost:8 --mpi-args=--oversubscribe numactl --interleave=all -- python -u main.py --dataset_path /data/converted --amp --xla
```

- multi-GPU for DGX-1 and DGX-2:
```bash
horovodrun -np 8 -H localhost:8 --mpi-args=--oversubscribe numactl --interleave=all -- python -u main.py --dataset_path /data/converted --amp --xla
```
</details>
<details>
<summary><b>3. Provide a fully preprocessed dataset, saved in split binary files, and a Feature Specification yaml file</b></summary>
This is the option to choose if you want full control over preprocessing and/or want to preprocess data directly to the target format.

Your final output will need to contain a Feature Specification yaml describing data and file layout. 
For an example feature specification file, refer to `tests/feature_specs/criteo_f15.yaml`

For details, refer to the [BYO dataset overview section](#byo-dataset-functionality-overview).
</details>



#### Channel definitions and requirements

This model defines three channels:

- categorical, accepting an arbitrary number of features
- numerical, accepting an arbitrary number of features
- label, accepting a single feature


The training script expects two mappings:

- train
- test

For performance reasons:
* The only supported dataset type is split binary
* Splitting chunks into multiple files is not supported.
* Each categorical feature has to be provided in a separate chunk
* All numerical features have to be provided in a single chunk
* All numerical features have to appear in the same order in channel_spec and source_spec
* Only integer types are supported for categorical features
* Only float16 is supported for numerical features

#### BYO dataset constraints for the model

There are the following constraints of BYO dataset functionality for this model:
1. The performance of the model depends on the dataset size. Generally, the model should scale better for datasets containing more data points. For a smaller dataset, you might experience slower performance than the one reported for Criteo
2. Using other datasets might require tuning some hyperparameters (for example, learning rate, beta1 and beta2) to reach desired accuracy.
3. The optimized cuda interaction kernels for FP16 and TF32 assume that the number of categorical variables is smaller than WARP_SIZE=32 and embedding size is <=128

