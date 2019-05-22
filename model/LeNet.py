from keras import optimizers
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout

# Bulid model
def LeNet(in_shape, n_class):
    model = Sequential()
    model.add(Conv2D(8, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape = in_shape))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(rate=0.4))
    model.add(Dense(64, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(rate=0.4))
    model.add(Dense(n_class, activation = 'softmax', kernel_initializer='he_normal'))
    return model