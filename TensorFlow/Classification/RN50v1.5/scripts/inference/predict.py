import sys
import os
import argparse
import re

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image

import class2label

image_extensions = ('.jpg', '.jpeg', '.png', '.JPEG')

def get_files(directory):
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(image_extensions)]
    return image_files

def load_images(image_paths):
    
    image_width = 224
    image_height = 224
    channels = 3
    size = image_width, image_height
    
    for path in image_paths:
        img = Image.open( path )
        img.load()
        img = img.resize((image_width, image_height), Image.ANTIALIAS)
        data = np.asarray( img, dtype="int32" )
        if np.shape(data) == (image_width, image_height, channels):
            data = np.transpose(data, (1, 0, 2))
            data = np.expand_dims(data, 0)
            yield path, data

def predict(model_path, file=None, directory=None):

    image_paths = []
    if file:
        image_paths.append(file)
    if directory:
        image_paths.extend(get_files(directory))
    
    image_paths.sort()
    
    images = []
    labels = []
    
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING], 
            model_path
        )
        file_writer = tf.summary.FileWriter('/images', sess.graph)
        i = 0;
        for path, image in load_images(image_paths):
            i += 1
            prediction = sess.run(
                'resnet50_v1.5/output/softmax:0',
                feed_dict={
                    'input_tensor:0': image
                }
            )
            
            label = class2label.class2label[np.argmax(prediction)]
            images.append(np.squeeze(image))
            labels.append(label)
            print(path, label)

def main(arguments):
    predict(arguments.model_path, arguments.file, arguments.directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for running prediction on RN50v15_TF SavedModel")
    
    parser.add_argument('-m', '--model_path', required=True, type=str, help="Directory containing exported SavedModel.")
    parser.add_argument('-f', '--file', required=False, default=None, type=str, help="Path to target file.")
    parser.add_argument('-d', '--directory', required=False, default=None, type=str, help="Path to target directory.")
    
    arguments, unparsed = parser.parse_known_args()
    main(arguments)
