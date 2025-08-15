import os
import tensorflow as tf
import cv2
from keras.preprocessing import image
import numpy as np
from PIL import Image
import base64
import requests
from keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.image import load_img
tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])


tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Model
    from keras.layers import Activation
    from keras.layers import BatchNormalization
    from keras.layers import Concatenate
    from keras.layers import Conv2D
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import GlobalAveragePooling2D
    from keras.layers import Input
    from keras.layers import Lambda
    from keras.layers import MaxPooling2D
    from keras.layers import add
    from keras import backend as K
else:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import Concatenate
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Lambda
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import add
    from tensorflow.keras import backend as K

# --------------------------------


def scaling(x, scale):
    return x * scale


def DNN(dimension=128):
    try:


        inputs = Input(shape=(160, 160, 3))
        x = Conv2D(32, 3, strides=2, padding='valid', use_bias=False, name='Conv2d_1a_3x3')(inputs)
        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_1a_3x3_BatchNorm')(x)

        x = Activation('relu', name='Conv2d_1a_3x3_Activation')(x)

        x = Conv2D(32, 3, strides=1, padding='valid', use_bias=False, name='Conv2d_2a_3x3')(x)

        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_2a_3x3_BatchNorm')(x)

        x = Activation('relu', name='Conv2d_2a_3x3_Activation')(x)

        x = Conv2D(64, 3, strides=1, padding='same', use_bias=False, name='Conv2d_2b_3x3')(x)

        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_2b_3x3_BatchNorm')(x)

        x = Activation('relu', name='Conv2d_2b_3x3_Activation')(x)

        x = MaxPooling2D(3, strides=2, name='MaxPool_3a_3x3')(x)

        x = Conv2D(80, 1, strides=1, padding='valid', use_bias=False, name='Conv2d_3b_1x1')(x)

        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_3b_1x1_BatchNorm')(x)

        x = Activation('relu', name='Conv2d_3b_1x1_Activation')(x)

        x = Conv2D(192, 3, strides=1, padding='valid', use_bias=False, name='Conv2d_4a_3x3')(x)

        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_4a_3x3_BatchNorm')(x)

        x = Activation('relu', name='Conv2d_4a_3x3_Activation')(x)

        x = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name='Conv2d_4b_3x3')(x)

        x = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False, name='Conv2d_4b_3x3_BatchNorm')(x)

        x = Activation('relu', name='Conv2d_4b_3x3_Activation')(x)

        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name='Block35_1_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block35_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name='Block35_1_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='Block35_1_Branch_1_Conv2d_0b_3x3')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_1_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_1_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name='Block35_1_Branch_2_Conv2d_0a_1x1')(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_1_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_1_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='Block35_1_Branch_2_Conv2d_0b_3x3')(
            branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_1_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_1_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='Block35_1_Branch_2_Conv2d_0c_3x3')(
            branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_1_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_1_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_1_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name='Block35_1_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_1_Activation')(x)

        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name='Block35_2_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_2_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block35_2_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name='Block35_2_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_2_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_2_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='Block35_2_Branch_1_Conv2d_0b_3x3')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_2_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_2_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name='Block35_2_Branch_2_Conv2d_0a_1x1')(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_2_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_2_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='Block35_2_Branch_2_Conv2d_0b_3x3')(
            branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_2_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_2_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='Block35_2_Branch_2_Conv2d_0c_3x3')(
            branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_2_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_2_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_2_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name='Block35_2_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_2_Activation')(x)

        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name='Block35_3_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_3_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block35_3_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name='Block35_3_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_3_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_3_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='Block35_3_Branch_1_Conv2d_0b_3x3')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_3_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_3_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name='Block35_3_Branch_2_Conv2d_0a_1x1')(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_3_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_3_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='Block35_3_Branch_2_Conv2d_0b_3x3')(
            branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_3_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_3_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='Block35_3_Branch_2_Conv2d_0c_3x3')(
            branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_3_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_3_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_3_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name='Block35_3_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_3_Activation')(x)

        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name='Block35_4_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_4_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block35_4_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name='Block35_4_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_4_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_4_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='Block35_4_Branch_1_Conv2d_0b_3x3')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_4_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_4_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name='Block35_4_Branch_2_Conv2d_0a_1x1')(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_4_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_4_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='Block35_4_Branch_2_Conv2d_0b_3x3')(
            branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_4_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_4_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='Block35_4_Branch_2_Conv2d_0c_3x3')(
            branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_4_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_4_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_4_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name='Block35_4_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_4_Activation')(x)

        branch_0 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name='Block35_5_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_5_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block35_5_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name='Block35_5_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_5_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_5_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='Block35_5_Branch_1_Conv2d_0b_3x3')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_5_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block35_5_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_2 = Conv2D(32, 1, strides=1, padding='same', use_bias=False, name='Block35_5_Branch_2_Conv2d_0a_1x1')(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_5_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_5_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='Block35_5_Branch_2_Conv2d_0b_3x3')(
            branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_5_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_5_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(32, 3, strides=1, padding='same', use_bias=False, name='Block35_5_Branch_2_Conv2d_0c_3x3')(
            branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block35_5_Branch_2_Conv2d_0c_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Block35_5_Branch_2_Conv2d_0c_3x3_Activation')(branch_2)
        branches = [branch_0, branch_1, branch_2]
        mixed = Concatenate(axis=3, name='Block35_5_Concatenate')(branches)
        up = Conv2D(256, 1, strides=1, padding='same', use_bias=True, name='Block35_5_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.17})(up)
        x = add([x, up])
        x = Activation('relu', name='Block35_5_Activation')(x)

        branch_0 = Conv2D(384, 3, strides=2, padding='valid', use_bias=False, name='Mixed_6a_Branch_0_Conv2d_1a_3x3')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Mixed_6a_Branch_0_Conv2d_1a_3x3_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Mixed_6a_Branch_0_Conv2d_1a_3x3_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name='Mixed_6a_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Mixed_6a_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, 3, strides=1, padding='same', use_bias=False, name='Mixed_6a_Branch_1_Conv2d_0b_3x3')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Mixed_6a_Branch_1_Conv2d_0b_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_0b_3x3_Activation')(branch_1)
        branch_1 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name='Mixed_6a_Branch_1_Conv2d_1a_3x3')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Mixed_6a_Branch_1_Conv2d_1a_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Mixed_6a_Branch_1_Conv2d_1a_3x3_Activation')(branch_1)
        branch_pool = MaxPooling2D(3, strides=2, padding='valid', name='Mixed_6a_Branch_2_MaxPool_1a_3x3')(x)
        branches = [branch_0, branch_1, branch_pool]
        x = Concatenate(axis=3, name='Mixed_6a')(branches)

        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_1_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_1_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False,
                          name='Block17_1_Branch_1_Conv2d_0b_1x7')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_1_Branch_1_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_1_Branch_1_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False,
                          name='Block17_1_Branch_1_Conv2d_0c_7x1')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_1_Branch_1_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_1_Branch_1_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_1_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name='Block17_1_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_1_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_2_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_2_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_2_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_2_Branch_2_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_2_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_2_Branch_2_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False,
                          name='Block17_2_Branch_2_Conv2d_0b_1x7')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_2_Branch_2_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_2_Branch_2_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False,
                          name='Block17_2_Branch_2_Conv2d_0c_7x1')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_2_Branch_2_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_2_Branch_2_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_2_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name='Block17_2_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_2_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_3_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_3_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_3_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_3_Branch_3_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_3_Branch_3_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_3_Branch_3_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False,
                          name='Block17_3_Branch_3_Conv2d_0b_1x7')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_3_Branch_3_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_3_Branch_3_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False,
                          name='Block17_3_Branch_3_Conv2d_0c_7x1')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_3_Branch_3_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_3_Branch_3_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_3_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name='Block17_3_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_3_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_4_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_4_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_4_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_4_Branch_4_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_4_Branch_4_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_4_Branch_4_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False,
                          name='Block17_4_Branch_4_Conv2d_0b_1x7')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_4_Branch_4_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_4_Branch_4_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False,
                          name='Block17_4_Branch_4_Conv2d_0c_7x1')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_4_Branch_4_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_4_Branch_4_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_4_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name='Block17_4_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_4_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_5_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_5_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_5_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_5_Branch_5_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_5_Branch_5_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_5_Branch_5_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False,
                          name='Block17_5_Branch_5_Conv2d_0b_1x7')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_5_Branch_5_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_5_Branch_5_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False,
                          name='Block17_5_Branch_5_Conv2d_0c_7x1')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_5_Branch_5_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_5_Branch_5_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_5_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name='Block17_5_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_5_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_6_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_6_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_6_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_6_Branch_6_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_6_Branch_6_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_6_Branch_6_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False,
                          name='Block17_6_Branch_6_Conv2d_0b_1x7')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_6_Branch_6_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_6_Branch_6_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False,
                          name='Block17_6_Branch_6_Conv2d_0c_7x1')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_6_Branch_6_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_6_Branch_6_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_6_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name='Block17_6_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_6_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_7_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_7_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_7_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_7_Branch_7_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_7_Branch_7_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_7_Branch_7_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False,
                          name='Block17_7_Branch_7_Conv2d_0b_1x7')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_7_Branch_7_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_7_Branch_7_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False,
                          name='Block17_7_Branch_7_Conv2d_0c_7x1')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_7_Branch_7_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_7_Branch_7_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_7_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name='Block17_7_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_7_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_8_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_8_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_8_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_8_Branch_8_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_8_Branch_8_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_8_Branch_8_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False,
                          name='Block17_8_Branch_8_Conv2d_0b_1x7')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_8_Branch_8_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_8_Branch_8_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False,
                          name='Block17_8_Branch_8_Conv2d_0c_7x1')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_8_Branch_8_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_8_Branch_8_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_8_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name='Block17_8_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_8_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_9_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_9_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_9_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_9_Branch_9_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_9_Branch_9_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_9_Branch_9_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False,
                          name='Block17_9_Branch_9_Conv2d_0b_1x7')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_9_Branch_9_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_9_Branch_9_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False,
                          name='Block17_9_Branch_9_Conv2d_0c_7x1')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_9_Branch_9_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_9_Branch_9_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_9_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name='Block17_9_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_9_Activation')(x)

        branch_0 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_10_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_10_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block17_10_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(128, 1, strides=1, padding='same', use_bias=False, name='Block17_10_Branch_10_Conv2d_0a_1x1')(
            x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_10_Branch_10_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_10_Branch_10_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(128, [1, 7], strides=1, padding='same', use_bias=False,
                          name='Block17_10_Branch_10_Conv2d_0b_1x7')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_10_Branch_10_Conv2d_0b_1x7_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_10_Branch_10_Conv2d_0b_1x7_Activation')(branch_1)
        branch_1 = Conv2D(128, [7, 1], strides=1, padding='same', use_bias=False,
                          name='Block17_10_Branch_10_Conv2d_0c_7x1')(branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block17_10_Branch_10_Conv2d_0c_7x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block17_10_Branch_10_Conv2d_0c_7x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block17_10_Concatenate')(branches)
        up = Conv2D(896, 1, strides=1, padding='same', use_bias=True, name='Block17_10_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.1})(up)
        x = add([x, up])
        x = Activation('relu', name='Block17_10_Activation')(x)

        branch_0 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name='Mixed_7a_Branch_0_Conv2d_0a_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Mixed_7a_Branch_0_Conv2d_0a_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Mixed_7a_Branch_0_Conv2d_0a_1x1_Activation')(branch_0)
        branch_0 = Conv2D(384, 3, strides=2, padding='valid', use_bias=False, name='Mixed_7a_Branch_0_Conv2d_1a_3x3')(
            branch_0)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Mixed_7a_Branch_0_Conv2d_1a_3x3_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Mixed_7a_Branch_0_Conv2d_1a_3x3_Activation')(branch_0)
        branch_1 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name='Mixed_7a_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Mixed_7a_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Mixed_7a_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name='Mixed_7a_Branch_1_Conv2d_1a_3x3')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Mixed_7a_Branch_1_Conv2d_1a_3x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Mixed_7a_Branch_1_Conv2d_1a_3x3_Activation')(branch_1)
        branch_2 = Conv2D(256, 1, strides=1, padding='same', use_bias=False, name='Mixed_7a_Branch_2_Conv2d_0a_1x1')(x)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Mixed_7a_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_0a_1x1_Activation')(branch_2)
        branch_2 = Conv2D(256, 3, strides=1, padding='same', use_bias=False, name='Mixed_7a_Branch_2_Conv2d_0b_3x3')(
            branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Mixed_7a_Branch_2_Conv2d_0b_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_0b_3x3_Activation')(branch_2)
        branch_2 = Conv2D(256, 3, strides=2, padding='valid', use_bias=False, name='Mixed_7a_Branch_2_Conv2d_1a_3x3')(
            branch_2)
        branch_2 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Mixed_7a_Branch_2_Conv2d_1a_3x3_BatchNorm')(branch_2)
        branch_2 = Activation('relu', name='Mixed_7a_Branch_2_Conv2d_1a_3x3_Activation')(branch_2)
        branch_pool = MaxPooling2D(3, strides=2, padding='valid', name='Mixed_7a_Branch_3_MaxPool_1a_3x3')(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = Concatenate(axis=3, name='Mixed_7a')(branches)

        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name='Block8_1_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_1_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_1_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name='Block8_1_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_1_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_1_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False,
                          name='Block8_1_Branch_1_Conv2d_0b_1x3')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_1_Branch_1_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_1_Branch_1_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False,
                          name='Block8_1_Branch_1_Conv2d_0c_3x1')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_1_Branch_1_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_1_Branch_1_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_1_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name='Block8_1_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_1_Activation')(x)

        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name='Block8_2_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_2_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_2_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name='Block8_2_Branch_2_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_2_Branch_2_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_2_Branch_2_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False,
                          name='Block8_2_Branch_2_Conv2d_0b_1x3')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_2_Branch_2_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_2_Branch_2_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False,
                          name='Block8_2_Branch_2_Conv2d_0c_3x1')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_2_Branch_2_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_2_Branch_2_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_2_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name='Block8_2_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_2_Activation')(x)

        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name='Block8_3_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_3_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_3_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name='Block8_3_Branch_3_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_3_Branch_3_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_3_Branch_3_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False,
                          name='Block8_3_Branch_3_Conv2d_0b_1x3')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_3_Branch_3_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_3_Branch_3_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False,
                          name='Block8_3_Branch_3_Conv2d_0c_3x1')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_3_Branch_3_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_3_Branch_3_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_3_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name='Block8_3_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_3_Activation')(x)

        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name='Block8_4_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_4_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_4_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name='Block8_4_Branch_4_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_4_Branch_4_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_4_Branch_4_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False,
                          name='Block8_4_Branch_4_Conv2d_0b_1x3')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_4_Branch_4_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_4_Branch_4_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False,
                          name='Block8_4_Branch_4_Conv2d_0c_3x1')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_4_Branch_4_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_4_Branch_4_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_4_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name='Block8_4_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_4_Activation')(x)

        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name='Block8_5_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_5_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_5_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name='Block8_5_Branch_5_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_5_Branch_5_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_5_Branch_5_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False,
                          name='Block8_5_Branch_5_Conv2d_0b_1x3')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_5_Branch_5_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_5_Branch_5_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False,
                          name='Block8_5_Branch_5_Conv2d_0c_3x1')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_5_Branch_5_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_5_Branch_5_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_5_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name='Block8_5_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 0.2})(up)
        x = add([x, up])
        x = Activation('relu', name='Block8_5_Activation')(x)

        branch_0 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name='Block8_6_Branch_0_Conv2d_1x1')(x)
        branch_0 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_6_Branch_0_Conv2d_1x1_BatchNorm')(branch_0)
        branch_0 = Activation('relu', name='Block8_6_Branch_0_Conv2d_1x1_Activation')(branch_0)
        branch_1 = Conv2D(192, 1, strides=1, padding='same', use_bias=False, name='Block8_6_Branch_1_Conv2d_0a_1x1')(x)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_6_Branch_1_Conv2d_0a_1x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_6_Branch_1_Conv2d_0a_1x1_Activation')(branch_1)
        branch_1 = Conv2D(192, [1, 3], strides=1, padding='same', use_bias=False,
                          name='Block8_6_Branch_1_Conv2d_0b_1x3')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_6_Branch_1_Conv2d_0b_1x3_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_6_Branch_1_Conv2d_0b_1x3_Activation')(branch_1)
        branch_1 = Conv2D(192, [3, 1], strides=1, padding='same', use_bias=False,
                          name='Block8_6_Branch_1_Conv2d_0c_3x1')(
            branch_1)
        branch_1 = BatchNormalization(axis=3, momentum=0.995, epsilon=0.001, scale=False,
                                      name='Block8_6_Branch_1_Conv2d_0c_3x1_BatchNorm')(branch_1)
        branch_1 = Activation('relu', name='Block8_6_Branch_1_Conv2d_0c_3x1_Activation')(branch_1)
        branches = [branch_0, branch_1]
        mixed = Concatenate(axis=3, name='Block8_6_Concatenate')(branches)
        up = Conv2D(1792, 1, strides=1, padding='same', use_bias=True, name='Block8_6_Conv2d_1x1')(mixed)
        up = Lambda(scaling, output_shape=K.int_shape(up)[1:], arguments={'scale': 1})(up)
        x = add([x, up])

        x = GlobalAveragePooling2D(name='AvgPool')(x)
        x = Dropout(1.0 - 0.8, name='Dropout')(x)

        x = Dense(dimension, use_bias=False, name='Bottleneck')(x)
        x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name='Bottleneck_BatchNorm')(x)


        model = Model(inputs, x, name='inception_resnet_v1')

        return model
    except Exception as e:
        raise e


def get_image_names(folder_path):
    image_names = []

    # Check if the provided path is a valid directory
    if os.path.isdir(folder_path):
        # Get all file names in the directory
        file_names = os.listdir(folder_path)

        # Filter out non-image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
        image_files = [file_name for file_name in file_names if os.path.splitext(file_name)[1] in image_extensions]

        # Get the full image paths and append them to the list
        image_names = [os.path.join(folder_path, file_name) for file_name in image_files]

    else:
        print("Invalid directory path.")

    return image_names

def load_facenet512():
    model = DNN(dimension=512)
    model.load_weights("facenet512_weights.h5")

    # img_path1 = r"E:\Face Verification\senet-face\data\Test_Live\200-test-face\6450243627_000.jpg"
    # img_path2 = r"E:\Face Verification\senet-face\data\Test_Live\200-test-face\6600032510_000.jpg"
    # img = image.load_img(img_path1, target_size=(160, 160))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    # embed1 = model.predict(x)
    #
    # img2 = image.load_img(img_path2, target_size=(160, 160))
    # y = image.img_to_array(img2)
    # y = np.expand_dims(y, axis=0)
    # y = preprocess_input(y)
    # embed2 = model.predict(y)
    #
    # img1_representation = represent(img_path=img_path1
    #                                      , is_face=False
    #                                      )
    #
    # img2_representation = represent(img_path=img_path2
    #                                      , is_face=False
    #                                      )
    #
    # print(embed1)

    folder_path = r"E:\Face Verification\senet-face\data\Test_Live\liveFrame_test"
    if os.path.isdir(folder_path):
        # Get all file names in the directory
        file_names = os.listdir(folder_path)
        for file in file_names:
            print(file)
            new_path = os.path.join(folder_path, file)
            frames = get_image_names(new_path)
            # print(frames[0][-18:-8])   frames[0][-18:-8]
            for idx, frame in enumerate(frames):
                sabt_ahval = os.path.join(r"E:\Face Verification\senet-face\data\Test_Live\sabtAhval_test", frames[0][-18:-8] + "_000.jpg")
                print(sabt_ahval)
                img = image.load_img(sabt_ahval, target_size=(160, 160))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                embed1 = model.predict(x)

                img2 = image.load_img(frame, target_size=(160, 160))
                y = image.img_to_array(img2)
                y = np.expand_dims(y, axis=0)
                y = preprocess_input(y)
                embed2 = model.predict(y)

                cosine_sim1 = findCosineDistance(embed1, embed2)
                print("Cosine Similarity:", cosine_sim1[0][0])
            # print("res_verification", res_verification)
            # if len(res_verification['verified']) > 0:
            #     print("Final Result", max(res_verification['verified'],key=res_verification['verified'].count))

    # Calculate cosine similarity
    # embedding1 = model.predict(img1_representation, verbose=0)[0].tolist()
    # embedding2 = model.predict(img2_representation, verbose=0)[0].tolist()
    # cosine_sim = findCosineDistance(embedding1, embedding2)
    # print("Cosine Similarity:", cosine_sim)
    # cosine_sim1 = findCosineDistance(embed1, embed2)
    # print("Cosine Similarity:", cosine_sim1[0][0])
    #
    # # Calculate Euclidean similarity
    # euclidean_sim = findEuclideanDistance(embedding1 , embedding2)
    # print("Euclidean Similarity:", euclidean_sim)
    # euclidean_sim = findEuclideanDistance(embed1, embed2)
    # print("Euclidean Similarity:", euclidean_sim)


    # return model


# def preprocess(img_path1):
#     img = image.load_img(img_path1, target_size=(160, 160))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def represent(img_path,
              face_detector=None,
              enforce_detection=True, align=True, normalization='base', is_face=False):

    # input_shape_x, input_shape_y =

    img = preprocess_face(img=img_path
                                    , target_size=(160, 160)
                                    , enforce_detection=enforce_detection

                                    , face_detector=face_detector
                                    , align=align, is_face=False)

    img = normalize_input(img=img, normalization=normalization)



    return img

def preprocess_face(img, target_size=(224, 224), grayscale=False, enforce_detection=True, detector_backend='opencv',
                    face_detector=None, return_region=False, align=True, is_face=False):


    if not is_face:
        img = load_image(img)
    base_img = img.copy()

    # img, region = detect_face(img=img
    #                           ,detector_backend=detector_backend
    #                           ,face_detector=face_detector
    #                           ,grayscale=grayscale
    #                           ,enforce_detection=enforce_detection
    #                           ,align=align)

    # --------------------------

    if img.shape[0] == 0 or img.shape[1] == 0:
        if enforce_detection == True:
            raise ValueError("Detected face shape is ", img.shape,
                             ". Consider to set enforce_detection argument to False.")
        else:  # restore base image
            img = base_img.copy()

    # --------------------------


    if grayscale == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------

    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        img = cv2.resize(img, dsize)

        # Then pad the other side to the target size by adding black pixels
        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        if grayscale == False:
            # Put the base image in the middle of the padded image
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)),
                         'constant')
        else:
            img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

    # ------------------------------------------


    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    # ---------------------------------------------------


    img_pixels = image.img_to_array(img)  # what this line doing? must?
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255  # normalize input in [0, 1]

    # ---------------------------------------------------

    if return_region == True:
        return img_pixels, region
    else:
        return img_pixels

def load_image(img):
    exact_image = False;
    base64_img = False;
    url_img = False

    if type(img).__module__ == np.__name__:
        exact_image = True

    elif len(img) > 11 and img[0:11] == "data:image/":
        base64_img = True

    elif len(img) > 11 and img.startswith("http"):
        url_img = True

    # ---------------------------

    if base64_img == True:
        img = loadBase64Img(img)

    elif url_img:
        img = np.array(Image.open(requests.get(img, stream=True).raw).convert('RGB'))

    elif exact_image != True:  # image path passed as input
        if os.path.isfile(img) != True:
            raise ValueError("Confirm that ", img, " exists")

        img = cv2.imread(img)

    return img
def loadBase64Img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
def normalize_input(img, normalization='base'):


    if normalization == 'base':
        return img
    else:


        img *= 255

        if normalization == 'raw':
            pass

    return img

def find_input_shape(model):
    # face recognition models have different size of inputs

    input_shape = model.layers[0].input_shape

    if type(input_shape) == list:
        input_shape = input_shape[0][1:3]
    else:
        input_shape = input_shape[1:3]

    # ----------------------

    if tf_major_version == 2 and tf_minor_version >= 5:
        x = input_shape[0];
        y = input_shape[1]
        input_shape = (y, x)

    # ----------------------

    if type(input_shape) == list:
        input_shape = tuple(input_shape)

    return input_shape
def detect_face(img, detector_backend='opencv', face_detector=None, grayscale=False, enforce_detection=True,
                align=True):

    img_region = [0, 0, img.shape[0], img.shape[1]]

    # ----------------------------------------------

    if detector_backend == 'skip':
        return img, img_region

    # ----------------------------------------------

    detector = FaceDetector()

    try:
        detected_face, img_region = detector.detect_face(face_detector=face_detector,
                                                             detector_backend=detector_backend,
                                                             img=img,
                                                             align=align)

    except Exception as e:
        print(e)
        detected_face = None

    if (isinstance(detected_face, np.ndarray)):
        return detected_face, img_region
    else:
        if detected_face == None:
            if enforce_detection != True:
                return img, img_region
            else:
                raise ValueError(
                    "Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")




def main():
    load_facenet512()

if __name__ == '__main__':

    main()