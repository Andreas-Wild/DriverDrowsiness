#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    13-Oct-2024 17:58:12

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    imageinput_unnormalized = keras.Input(shape=(64,64,1), name="imageinput_unnormalized")
    imageinput = keras.layers.Normalization(axis=(1,2,3), name="imageinput_")(imageinput_unnormalized)
    conv_1 = layers.Conv2D(8, (5,5), strides=(2,2), name="conv_1_")(imageinput)
    relu_1 = layers.ReLU()(conv_1)
    maxpool_1 = layers.MaxPool2D(pool_size=(2,2), strides=(1,1))(relu_1)
    conv_2 = layers.Conv2D(16, (3,3), strides=(2,2), name="conv_2_")(maxpool_1)
    relu_2 = layers.ReLU()(conv_2)
    maxpool_2 = layers.MaxPool2D(pool_size=(2,2), strides=(1,1))(relu_2)
    fc_1 = layers.Reshape((-1,), name="fc_1_preFlatten1")(maxpool_2)
    fc_1 = layers.Dense(32, name="fc_1_")(fc_1)
    relu_3 = layers.ReLU()(fc_1)
    dropout = layers.Dropout(0.500000)(relu_3)
    fc_2 = layers.Dense(1, name="fc_2_")(dropout)
    layer = layers.Activation('sigmoid')(fc_2)

    model = keras.Model(inputs=[imageinput_unnormalized], outputs=[layer])
    return model
