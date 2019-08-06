import numpy as np
import skimage

def load_image(image_path):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    img = skimage.img_as_float(skimage.io.imread(image_path))
    if len(img.shape) == 2:
        img = np.array([img, img, img]).swapaxes(0,2)
    return img

def rescale(img, input_height, input_width):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    aspect = img.shape[1]/float(img.shape[0])
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    return imgScaled

def crop_center(img,cropx,cropy):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def normalize(img, mean=128, std=128):
    img = (img * 256 - mean) / std
    return img

def prepare_input(img_uri):
    img = load_image(img_uri)
    img = rescale(img, 300, 300)
    img = crop_center(img, 300, 300)
    img = normalize(img)

    return img
