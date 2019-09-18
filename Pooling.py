# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:06:31 2019

@author: lawle
"""

import keras.backend as K
import numpy as np
from keras.layers.pooling import _GlobalPooling2D
from keras.layers import Activation
import tensorflow as tf


class GeMPooling2D(_GlobalPooling2D):
    """Generallized Mean pooling operation for spatial data.
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """
    
    def call(self, inputs):
        self.p = tf.Variable(3, dtype = tf.float32, trainable = True,name = "GeM_p")
        if self.data_format == 'channels_last':
            W = tf.Variable(np.random.rand(1,inputs.shape[1]*inputs.shape[2]),dtype = tf.float32)
            W = Activation("softmax")(W)
            W = tf.reshape(W,(inputs.shape[1],inputs.shape[2],1,1)) 
            tile_W = tf.tile(W, [1,1,inputs.shape[3],1])
            x = K.pow(inputs,self.p)
            output = tf.nn.depthwise_conv2d(x, tile_W, strides=[1, 1, 1, 1], padding='VALID')
            output = tf.reshape(output, (-1,inputs.shape[3]))
            return K.pow(output, 1/self.p)
        else:
            return K.pow(K.mean(K.pow(inputs,self.p), axis = [2,3]), 1/self.p)
        
