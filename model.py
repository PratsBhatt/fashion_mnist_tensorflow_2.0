import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model


def create_model(input_shape, num_classes):
    i = Input(shape = input_shape)
    x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
    x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(i, x)