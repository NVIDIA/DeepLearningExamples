## Quick Start Guide

To prepare the Criteo 1TB dataset for training, follow these steps.  

1. Make sure you meet the prerequisites.

You will need around 4TB of storage for storing the original Criteo 1TB dataset, the results of some
intermediate preprocessing steps and the final dataset. The final dataset itself will take about 400GB.

We recommend using local storage, such as a fast SSD drive, to run the preprocessing. Using other types of storage
will negatively impact the preprocessing time.


2. Build the preprocessing docker image.
```bash
docker build -t preproc_docker_image -f Dockerfile_spark .
```

3. Download the data by following the instructions at: http://labs.criteo.com/2013/12/download-terabyte-click-logs/.

When you have successfully downloaded the dataset, put it in the `/data/criteo_orig` directory in the container
(`$PWD/data/criteo_orig` in the host system).

4. Start an interactive session in the NGC container to run preprocessing.
The DLRM TensorFlow container can be launched with:

```bash
mkdir -p data
docker run --runtime=nvidia -it --rm --ipc=host  -v ${PWD}/data:/data preproc_docker_image bash
```

5. Unzip the data with:

```bash
gunzip /data/criteo_orig/*.gz
```

6. Preprocess the data.

Here are a few examples of different preprocessing commands.
For the details on how those scripts work and a detailed description of all the parameters,
consult the [preprocess with spark section](criteo_dataset.md#preprocess-with-spark).

```bash
export download_dir=/data/criteo_orig
export final_output_dir=/data/preprocessed

cd preproc

# to run on a DGX-2 with a frequency limit of 3 (will need 8xV100-32GB to fit the model in GPU memory)
./prepare_dataset.sh DGX2 3

# to run on a DGX-2 with a frequency limit of 15 (should fit on a single V100-32GB):
./prepare_dataset.sh DGX2 15

# to run on CPU with a frequency limit of 15:
./prepare_dataset.sh CPU 15

# to run on DGX-2 with no frequency limit:
./prepare_dataset.sh DGX2 0
```

7. Verify the preprocessed data

After running `tree /data/preprocessed` you should see the following directory structure:
```bash
$ tree /data/preprocessed
/data/preprocessed
├── feature_spec.yaml
├── test
│   ├── cat_0.bin
│   ├── cat_1.bin
│   ├── ...
│   ├── label.bin
│   └── numerical.bin
└── train
    ├── cat_0.bin
    ├── cat_1.bin
    ├── ...
    ├── label.bin
    └── numerical.bin

2 directories, 57 files
```


## Advanced

### Dataset guidelines

The first 23 days are used as the training set. The last day is split in half.
The first part is used as a validation set and the second set is used as a hold-out test set.

The preprocessing steps applied to the raw data include:
- Replacing the missing values with `0`.
- Replacing the categorical values that exist fewer than 15 times with a special value.
- Converting the hash values to consecutive integers.
- Adding 2 to all the numerical features so that all of them are greater or equal to 1.
- Taking a natural logarithm of all numerical features.


### Preprocess with Spark

The preprocessing scripts provided in this repository support running both on CPU and on DGX-2 using [Apache Spark 3.0](https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/apache-spark-3/).
It should be possible to change the values in `preproc/dgx2_config.sh`
so that they'll work on other hardware platforms such as DGX-1.

Note that the preprocessing will require about 4TB of disk storage.

The syntax for the preprocessing script is as follows:
```bash
cd preproc
./prepare_dataset.sh <DGX2|CPU> <frequency_threshold>
```

The first argument is the hardware platform to use (either DGX-2 or pure-CPU). The second argument means the frequency
threshold to apply to the categorical variables. For a frequency threshold `T`, the categorical values that occur less
often than `T` will be replaced with a special embedding. Thus, a larger value of `T` will require smaller embedding tables
and will substantially reduce the overall size of the model.

For the Criteo Terabyte dataset we recommend a frequency threshold of `T=3` if you intend to run the hybrid-parallel mode
on multiple GPUs. If you want to make the model fit into a single NVIDIA Tesla V100-32GB, you can set `T=15`.

The preprocessing scripts makes use of the following environment variables to configure the data directory paths:
- `download_dir` – this directory should contain the original Criteo Terabyte CSV files
- `spark_output_path` – directory to which the parquet data will be written
- `conversion_intermediate_dir` – directory used for storing intermediate data used to convert from parquet to train-ready format
- `final_output_dir` – directory to store the final results of the preprocessing which can then be used to train DLRM

The script `spark_data_utils.py` is a PySpark application, which is used to preprocess the Criteo Terabyte Dataset. In the Docker image, we have installed Spark 3.0.1, which will start a standalone cluster of Spark. The scripts `run_spark_cpu.sh` and `run_spark_gpu.sh` start Spark, then runs several PySpark jobs with `spark_data_utils.py`, for example:
generates the dictionary
- transforms the train dataset
- transforms the test dataset
- transforms the validation dataset

    Change the variables in the `run-spark.sh` script according to your environment.
    Configure the paths.
```
export SPARK_LOCAL_DIRS=/data/spark-tmp
export INPUT_PATH=/data/criteo
export OUTPUT_PATH=/data/output
```
Note that the Spark job requires about 3TB disk space used for data shuffle.

Where:
`SPARK_LOCAL_DIRS` is the path where Spark uses to write shuffle data.
`INPUT_PATH` is the path of the Criteo Terabyte Dataset, including uncompressed files like day_0, day_1…
`OUTPUT_PATH` is where the script writes the output data. It will generate the following subdirectories of `models`, `train`, `test`, and `validation`.
- The `model` is the dictionary folder.
- The `train` is the train dataset transformed from day_0 to day_22.
- The `test` is the test dataset transformed from the prior half of day_23.
- The `validation` is the dataset transformed from the latter half of day_23.

Configure the resources which Spark will use.
```
export TOTAL_CORES=80
export TOTAL_MEMORY=800
```

Where:
`TOTAL_CORES` is the total CPU cores you want Spark to use.

`TOTAL_MEMORY` is the total memory Spark will use.

Configure frequency limit.
```
USE_FREQUENCY_LIMIT=15
```
The frequency limit is used to filter out the categorical values which appear less than n times in the whole dataset, and make them be 0. Change this variable to 1 to enable it. The default frequency limit is 15 in the script. You also can change the number as you want by changing the line of `OPTS="--frequency_limit 8"`.

