ImageNet Dataset Build Scripts
==========================

What is this?
------------------

This directory includes all the scripts necessary to build the ImageNet dataset in a sharded protobuf representation, which is recommended by the TensorFlow team for performance reasons. The protobuf files will include a large number of JPEG images in one file, along with the image metadata (image class, bounding boxes, etc.). This will ensure good performance on both SSDs and magnetic hard drives. The protobufs will contain TFRecord data types, which are standard for TensorFlow.


Performance considerations
----------------------------------------

This script is largely based on TensorFlow's ImageNet preprocessing script for the Inception v3 model (see [here](https://github.com/tensorflow/models/tree/master/inception/inception/data)). The advantages of the dataset preprocessed in this fashion are discussed here.

 1. Protobufs containing many JPEGs are much faster to process than reading raw JPEGs, especially on magnetic disks, by avoiding seek time. This also tends to help on SSDs, because of sequential reads, which are still a bit faster than random reads. This was the case in the original TensorFlow preprocessing script.
 2. This version of the preprocessing scripts is independent of [Bazel](https://bazel.build/), Google's build tool. The Bazel requirement to "build" the Python and shell scripts is unnecessary and is a heavy-weight step that can has been avoided here. The scripts work the same way as the public Google scripts, but one can run them immediately without needing to set up Bazel and going through incantations like ```bazel build inception/download_and_preprocess_imagenet``` (see [here](https://github.com/tensorflow/models/tree/master/inception)). This is a modification to the original Google script.
 3. Pre-resizing while building the dataset is essential for good performance.The speedup from uniformly-sized images can be significant relative to original ImageNet, while running AlexNet-OWT (the gains are smaller for compute-heavy models such as Inception v3 and ResNet-50). Note that the current implementation does not preserve the aspect ratio while creating uniform-sized images. An alternative would be to resize while preserving the aspect ratio, then crop. This script is meant to help the user with preprocessing the model for performance reasons, but it may need to be tweaked to provide the best machine learning results. It's open-source so the user can modify it to their liking. This is a modification to the original Google script.
 4. Efficient storage - the original JPEGs are stored with a [quality factor](https://en.wikipedia.org/wiki/JPEG) of 100, but the color distortions tend not to happen until Q drops below 85. These scripts store the images with a Q factor of 90 by default, which reduces the image size by 75% while causing minimal color distortions of no consequence for convolutional neural network training. The impact of this is very significant due to a reduction in I/O load, particularly in a multi-GPU setting, when more total disk accesses need to take place to feed more than one GPU. This is a modification to the original Google script.

----------

How to run the scripts?
--------------------------------

 1. Create an ImageNet account at http://image-net.org. You will need a user ID
    and the access key provided upon registration.
 2. Run the download-imagenet.sh script. You will be asked for your ImageNet user ID, ImageNet password, and the directory in which to store the dataset. Future re-running of this script will be optimized in that if the tarballs containing the dataset are already available in the target directory, they won't be re-downloaded. However, the unzipping of the tarballs will still take place, so if you already ran this script, don't run it again.
 3. Run the generate_tfrecord_protos.sh script. You will be asked about the location of the files downloaded in step 2, as well as the directory in which to store the protobuf files to be used by TensorFlow. Additional questions will pertain to whether original or pre-resized images are to be stored (it is strongly recommended that pre-resizing be chosen), the height and width of the images after resizing (if pre-resizing is chosen), and the JPEG Q factor (Q=90 is recommended).

> **Note:**

> Running this script requires a lot of memory. Make sure you have at least 16 GB RAM free on your machine, and preferably 32 GB. This is due to a bug in TensorFlow that is currently being investigaged. It has nothing to do with the script, rather with TensorFlow core. It shows up only in case of datasets with millions of files. Once the ImageNet dataset size is reduced to a few thousand files (as it will be after these scripts are run, due to storage of many JPEGs in one protobuf), we could no longer replicate it while training models.

> If the script fails, examine the dmesg output - it likely failed due to an out of memory error. If that was the case, try freeing up memory. There is an included Python script called purge_mem_caches.py, which can be run on the host. This usually helps fix things that can't be resolved by just shutting down applications, such as purging virtual memory pages left by previous runs of TensorFlow, that weren't cleaned up by either the application or the OS. Note that running this script in the container itself, rather than the host, won't have any effect. Note also that this script has a dependency on psutil and pexpect, Python packages that can be installed using pip.gg
