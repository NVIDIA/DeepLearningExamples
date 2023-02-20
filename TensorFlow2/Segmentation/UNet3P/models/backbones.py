"""
Unet3+ backbones
"""
import tensorflow as tf
import tensorflow.keras as k
from .unet3plus_utils import conv_block


def vgg16_backbone(input_layer, ):
    """ VGG-16 backbone as encoder for UNet3P """

    base_model = tf.keras.applications.VGG16(
        input_tensor=input_layer,
        weights=None,
        include_top=False
    )

    # block 1
    e1 = base_model.get_layer("block1_conv2").output  # 320, 320, 64
    # block 2
    e2 = base_model.get_layer("block2_conv2").output  # 160, 160, 128
    # block 3
    e3 = base_model.get_layer("block3_conv3").output  # 80, 80, 256
    # block 4
    e4 = base_model.get_layer("block4_conv3").output  # 40, 40, 512
    # block 5
    e5 = base_model.get_layer("block5_conv3").output  # 20, 20, 512

    return [e1, e2, e3, e4, e5]


def vgg19_backbone(input_layer, ):
    """ VGG-19 backbone as encoder for UNet3P """

    base_model = tf.keras.applications.VGG19(
        input_tensor=input_layer,
        weights=None,
        include_top=False
    )

    # block 1
    e1 = base_model.get_layer("block1_conv2").output  # 320, 320, 64
    # block 2
    e2 = base_model.get_layer("block2_conv2").output  # 160, 160, 128
    # block 3
    e3 = base_model.get_layer("block3_conv4").output  # 80, 80, 256
    # block 4
    e4 = base_model.get_layer("block4_conv4").output  # 40, 40, 512
    # block 5
    e5 = base_model.get_layer("block5_conv4").output  # 20, 20, 512

    return [e1, e2, e3, e4, e5]


def unet3plus_backbone(input_layer, filters):
    """ UNet3+ own backbone """
    """ Encoder"""
    # block 1
    e1 = conv_block(input_layer, filters[0])  # 320*320*64
    # block 2
    e2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 160*160*64
    e2 = conv_block(e2, filters[1])  # 160*160*128
    # block 3
    e3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 80*80*128
    e3 = conv_block(e3, filters[2])  # 80*80*256
    # block 4
    e4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 40*40*256
    e4 = conv_block(e4, filters[3])  # 40*40*512
    # block 5, bottleneck layer
    e5 = k.layers.MaxPool2D(pool_size=(2, 2))(e4)  # 20*20*512
    e5 = conv_block(e5, filters[4])  # 20*20*1024

    return [e1, e2, e3, e4, e5]
