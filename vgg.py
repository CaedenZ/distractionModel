from keras.models import Sequential
from keras.optimizers import SGD,Adadelta
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D,AveragePooling2D
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
import keras

def VGG():
    img_rows, img_cols = 48, 48
    model = Sequential()
    model.add(Convolution2D(64, 5, 5, border_mode='valid',
                            input_shape=(img_rows, img_cols, 1)))
    model.add(PReLU(init='zero', weights=None))
    model.add(ZeroPadding2D(padding=(2, 2), dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3))
    model.add(PReLU(init='zero', weights=None))
    model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3))
    model.add(PReLU(init='zero', weights=None))

    model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))

    model.add(Dense(6))

    model.add(Activation('softmax'))

    ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=ada,
                  metrics=['accuracy'])
    
    return model


