from sklearn import preprocessing
import numpy as np
import h5py
from keras.utils import to_categorical

import keras
from keras.models import Sequential, Model
from keras.layers import GlobalMaxPooling2D, Dense, Activation, Conv2D, MaxPooling2D, Dropout, Input, AveragePooling2D, ZeroPadding2D, BatchNormalization
from keras.regularizers import l2


def fourier(datapoint):
    '''Fourier transform the xy data'''
    fdata = datapoint
    yf = np.fft.fft(datapoint[:, 1])
    # print(yf.shape)
    yf = yf[:len(yf)]
    fdata[:, 1] = yf

    return fdata


def regularise(data, ft):
    '''Correct for the Q^4 dependence and normalise data to between 0 and 1'''
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    reg_data = data
    for n in range(len(data)):
        datapoint = data[n]
        ycorrected = np.multiply(datapoint[:, 1], datapoint[:, 0] ** 4)
        datapoint[:, 1] = np.transpose(ycorrected)
        datapoint = min_max_scaler.fit_transform(datapoint)
        if ft is True: # Optional Fourier transform
            datapoint = fourier(datapoint)
        reg_data[n] = datapoint

    return reg_data


def noisy(xy):
    '''Add noise'''
    direct_beam = np.loadtxt('directbeam_noise.dat', delimiter=',')[:, 0:2]
    qs = xy[:, 0]
    flux_density = np.interp(qs, direct_beam[:, 0], direct_beam[:, 1])
    flux_density = flux_density * 100

    reflectance = xy[:, 1]
    out = xy
    noisydata = []
    for i, r in zip(flux_density, reflectance):
        incoming = np.random.poisson(i)
        if incoming == 0:
            noisydata.append(0.0)
        else:
            reflected = np.random.poisson(r * i)
            noisydata.append(reflected / float(incoming))
    out[:, 1] = np.transpose(noisydata)
    return out


def addbackground(xy, backgroundrate):
    '''Add background noise'''
    out = xy
    reflectance = xy[:, 1]
    backgroundydata = []
    for r in reflectance:
        reflected = r + np.random.normal(0.5, .4) * backgroundrate - np.random.normal(0.5, 0.4) * backgroundrate
        while reflected < 0:
            reflected = 0.0 + backgroundrate * np.random.normal(1, .2)
        backgroundydata.append(reflected)
    out[:, 1] = np.transpose(backgroundydata)
    return out


def loadin(fileloc, layernum, test_or_train):
    '''Loads in a given hdf5 file.

    Args: 
    fileloc -- file path for the hdf5 file
    layernum -- whether it's a 0, 1, or 2 layer system
    test_or_train -- testing or training data

    '''
    ml = h5py.File(fileloc, 'r')
    datafile = ml.get(test_or_train.upper())
    if layernum == 2:
        x = datafile[:, 0, 2:, :]
        y = datafile[:, 0, :2, :]
    elif layernum == 1:
        x = datafile[:, 0, 1:, :]
        y = datafile[:, 0, 0, :]
    else:
        x = datafile[:, 0, 3:, :]
        y = datafile[:, 0, 2, :]

    ml.close()

    return x, y


def preprocess_x(x_data, train_bool=True):
    # Ridiculously inefficient. Put timings in later to diagnose
    '''Packages the previous functions together'''

    x = x_data[:, :, :]  # cut off spike
    if train_bool == True:
        for n in range(len(x)):  # add noise
            x[n] = noisy(x[n])
            x[n] = addbackground(x[n], 5E-7)

    x = regularise(x, False)
    x = x.reshape(x.shape + (1,))

    return x


def categorical_to_d(one_hot):
    '''Neural net output to thickness'''
    label = np.argmax(one_hot) * 333
    return label


def categorical_to_sld(one_hot):
    '''Neural net output to density'''
    label = np.argmax(one_hot) * 0.1
    return label




# from FILE_HANDLING_CLOUD import *
class Layer:
    def __init__(self, thickness, density):
        self.d = thickness
        self.sld = density

    def get_d(self):
        return self.d

    def get_sld(self):
        return self.sld


class nn:
    def __init__(self):
        self.n_class_model = Sequential()
        self.multi_d_model = Sequential()
        self.multi_sld_model = Sequential()
        self.single_d_model = Sequential()
        self.single_sld_model = Sequential()
        self.d_sld_1_model = Sequential()

        self.n_class_history = None
        self.multi_d_history = None
        self.multi_sld_history = None
        self.single_d_history = None
        self.single_sld_history = None

        self.sld_bins = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                         0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25,
                         0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41,
                         0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57,
                         0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73,
                         0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
                         0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
                         0.99]  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.d_bins = [33.333333333333336, 66.66666666666667, 100.0, 133.33333333333334, 166.66666666666666, 200.0, 233.33333333333334,
                       266.6666666666667, 300.0, 333.3333333333333, 366.6666666666667, 400.0, 433.3333333333333, 466.6666666666667, 500.0,
                       533.3333333333334, 566.6666666666666, 600.0, 633.3333333333334, 666.6666666666666, 700.0, 733.3333333333334,
                       766.6666666666666, 800.0, 833.3333333333334, 866.6666666666666, 900.0, 933.3333333333334, 966.6666666666666, 1000.0,
                       1033.3333333333333, 1066.6666666666667, 1100.0, 1133.3333333333333, 1166.6666666666667, 1200.0, 1233.3333333333333,
                       1266.6666666666667, 1300.0, 1333.3333333333333, 1366.6666666666667, 1400.0, 1433.3333333333333, 1466.6666666666667,
                       1500.0, 1533.3333333333333, 1566.6666666666667, 1600.0, 1633.3333333333333, 1666.6666666666667, 1700.0, 1733.3333333333333,
                       1766.6666666666667, 1800.0, 1833.3333333333333, 1866.6666666666667, 1900.0, 1933.3333333333333, 1966.6666666666667, 2000.0,
                       2033.3333333333333, 2066.6666666666665, 2100.0, 2133.3333333333335, 2166.6666666666665, 2200.0, 2233.3333333333335,
                       2266.6666666666665, 2300.0, 2333.3333333333335, 2366.6666666666665, 2400.0, 2433.3333333333335, 2466.6666666666665,
                       2500.0, 2533.3333333333335, 2566.6666666666665, 2600.0, 2633.3333333333335, 2666.6666666666665, 2700.0, 2733.3333333333335,
                       2766.6666666666665, 2800.0, 2833.3333333333335, 2866.6666666666665, 2900.0, 2933.3333333333335, 2966.6666666666665, 3000.0]


        self.sample_1 = []  # possible samples, listed from most probable to least
        self.sample_2 = []
        self.sample_3 = []
        self.sample_4 = []

    def n_classification(self, x_train, y, batch_size, epochs, learning_rate, l2_lambda):
        '''The network that determines the number of layers.'''

        y_train = to_categorical(y[:, 0])

        input_shape = (None, 2, 1)

        tb = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

        self.n_class_model = Sequential()

        # zeropadding2d
        self.n_class_model.add(
            Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 2), strides=(1, 1), padding='same',
                   activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.n_class_model.add(MaxPooling2D(pool_size=(3, 1), padding='same'))
        self.n_class_model.add(BatchNormalization())
        self.n_class_model.add(Conv2D(filters=128, kernel_size=(3, 2), strides=(1, 1), padding='same',
                                      activation='relu', kernel_initializer='he_normal',
                                      kernel_regularizer=l2(l2_lambda)))
        self.n_class_model.add(MaxPooling2D(pool_size=(3, 1), padding='same'))
        self.n_class_model.add(BatchNormalization())
        self.n_class_model.add(Conv2D(filters=32, kernel_size=(3, 2), strides=(1, 1), padding='same',
                                      kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda),
                                      activation='relu'))
        self.n_class_model.add(MaxPooling2D(pool_size=(3, 1), padding='same'))
        self.n_class_model.add(BatchNormalization())
        self.n_class_model.add(
            Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.n_class_model.add(Dropout(0.25))
        self.n_class_model.add(GlobalMaxPooling2D())
        self.n_class_model.add(
            Dense(512, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.n_class_model.add(Dropout(0.25))
        self.n_class_model.add(
            Dense(1024, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.n_class_model.add(Dropout(0.25))
        self.n_class_model.add(
            Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.n_class_model.add(Dropout(0.25))

        self.n_class_model.add(Dense(3, activation='softmax'))

        self.n_class_model.compile(loss='categorical_crossentropy',
                                   optimizer=keras.optimizers.Adam(lr=learning_rate, epsilon=None, decay=0.0),
                                   metrics=['accuracy'])

        self.n_class_history = self.n_class_model.fit(x_train, y_train,
                                                      batch_size=batch_size,
                                                      epochs=epochs,
                                                      verbose=1, validation_split=0.1, callbacks=[tb])

        save = input('Save? y/n')
        if save == 'y':
            self.n_class_model.save('MODELS/N_CLASSIFICATION_NET.h5')

    def multi_d(self, x_train, y, batch_size, epochs, learning_rate, l2_lambda):

        y_train = y[:, :, 1]
        print(y_train)
        y_train = np.round(y_train * 3, -2) / 3
        y_train_dict = dict(zip(np.unique(y_train), range(len(np.unique(y_train)))))
        y_train = np.vectorize(y_train_dict.get)(y_train)
        y_train = to_categorical(y_train)
        y1_train = y_train[:, 0, :]
        y2_train = y_train[:, 1, :]
        print('Data loaded')
        input_shape = (None, 2, 1)

        tb = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

        main_input = Input(shape=input_shape)

        x = Conv2D(filters=16, kernel_size=(3, 2), strides=(1, 1), padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(main_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 1), padding='same')(x)

        x = Conv2D(filters=32, kernel_size=(3, 2), strides=(1, 1), padding='same'
                   , kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 1), padding='same', strides=(3, 1))(x)

        x = Conv2D(filters=64, kernel_size=(3, 2), strides=(1, 1), padding='same'
                   , kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        x = GlobalMaxPooling2D()(x)

        x1 = Dense(512, kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Dropout(0.25)(x1)
        x1 = Dense(1024, kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Dropout(0.25)(x1)
        x1 = Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x2 = Dense(512, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(x)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        x2 = Dropout(0.25)(x2)
        x2 = Dense(1024, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(x2)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        x2 = Dropout(0.25)(x2)
        x2 = Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(x2)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)

        x1 = Dense(90)(x1)
        x2 = Dense(90)(x2)

        output1 = Activation('softmax', name='layer_1_output')(x1)
        output2 = Activation('softmax', name='layer_2_output')(x2)

        self.multi_d_model = Model(input=main_input, outputs=[output1, output2])

        self.multi_d_model.compile(loss='categorical_crossentropy',
                                   optimizer=keras.optimizers.Adam(lr=learning_rate, epsilon=None, decay=0.0),
                                   metrics=['accuracy'], loss_weights=[0.5, 0.5])

        self.multi_d_history = self.multi_d_model.fit(x_train, [y1_train, y2_train],
                                                      batch_size=batch_size,
                                                      epochs=epochs,
                                                      verbose=1, validation_split=0.1, callbacks=[tb])
        save = input('Save? y/n')
        if save == 'y':
            self.multi_d_model.save('MODELS/2_LAYER_D_NET.h5')

    def multi_sld(self, x_train, y, batch_size, epochs, learning_rate, l2_lambda):
        # x_train = preprocess_x(x_train)
        y_train = y[:, :, 0]
        y_train = np.round(y_train, 2)
        y_train_dict = dict(zip(np.unique(y_train), range(len(np.unique(y_train)))))
        y_train = np.vectorize(y_train_dict.get)(y_train)
        y_train = to_categorical(y_train)
        y1_train = y_train[:, 0, :]
        y2_train = y_train[:, 1, :]

        input_shape = (None, 2, 1)

        tb = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                         write_graph=True, write_images=True)

        main_input = Input(shape=input_shape)

        x = Conv2D(filters=16, kernel_size=(3, 2), strides=(1, 1), padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(main_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 1), padding='same')(x)

        x = Conv2D(filters=32, kernel_size=(3, 2), strides=(1, 1), padding='same'
                   , kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 1), padding='same', strides=(3, 1))(x)

        x = Conv2D(filters=64, kernel_size=(3, 2), strides=(1, 1), padding='same'
                   , kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)

        x = GlobalMaxPooling2D()(x)

        x1 = Dense(512, kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Dropout(0.25)(x1)
        x1 = Dense(1024, kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Dropout(0.25)(x1)
        x1 = Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x2 = Dense(512, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(x)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        x2 = Dropout(0.25)(x2)
        x2 = Dense(1024, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(x2)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        x2 = Dropout(0.25)(x2)
        x2 = Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda))(x2)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)

        x1 = Dense(105)(x1)
        x2 = Dense(105)(x2)

        output1 = Activation('softmax', name='layer_1_output')(x1)
        output2 = Activation('softmax', name='layer_2_output')(x2)

        self.multi_sld_model = Model(input=main_input, outputs=[output1, output2])

        self.multi_sld_model.compile(loss='categorical_crossentropy',
                                     optimizer=keras.optimizers.Adam(lr=learning_rate, epsilon=None, decay=0.0),
                                     metrics=['accuracy'], loss_weights=[0.5, 0.5])

        self.multi_sld_history = self.multi_sld_model.fit(x_train, [y1_train, y2_train],
                                                          batch_size=batch_size,
                                                          epochs=epochs,
                                                          verbose=1, validation_split=0.1, callbacks=[tb])
        save = input('Save? y/n')
        if save == 'y':
            self.multi_sld_model.save('MODELS/2_LAYER_SLD_NET.h5')

    def single_d(self, x_train, y_train, batch_size, epochs, learning_rate, l2_lambda):
        y_train = y_train[:, 1]
        y_train = np.round(y_train * 3, -2) / 3
        y_train_dict = dict(zip(np.unique(y_train), range(len(np.unique(y_train)))))
        y_train = np.vectorize(y_train_dict.get)(y_train)
        y_train = to_categorical(y_train)

        input_shape = (None, 2, 1)
        l2_lambda = 0.0005

        self.single_d_model.add(
            Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 2), strides=(1, 1), padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.single_d_model.add(BatchNormalization())
        self.single_d_model.add(Activation('relu'))
        self.single_d_model.add(MaxPooling2D(pool_size=(3, 1), padding='same'))

        self.single_d_model.add(Conv2D(filters=128, kernel_size=(3, 2), strides=(1, 1), padding='same'
                                       , kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.single_d_model.add(BatchNormalization())
        self.single_d_model.add(Activation('relu'))
        self.single_d_model.add(MaxPooling2D(pool_size=(3, 1), padding='same'))

        self.single_d_model.add(Conv2D(filters=64, kernel_size=(3, 2), strides=(1, 1), padding='same'
                                       , kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.single_d_model.add(BatchNormalization())
        self.single_d_model.add(Activation('relu'))
        self.single_d_model.add(GlobalMaxPooling2D())

        self.single_d_model.add(Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.single_d_model.add(BatchNormalization())
        self.single_d_model.add(Activation('relu'))
        self.single_d_model.add(Dropout(0.15))
        self.single_d_model.add(Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.single_d_model.add(BatchNormalization())
        self.single_d_model.add(Activation('relu'))
        self.single_d_model.add(Dropout(0.15))
        self.single_d_model.add(Dense(64, kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.single_d_model.add(BatchNormalization())
        self.single_d_model.add(Activation('relu'))
        self.single_d_model.add(Dropout(0.15))

        self.single_d_model.add(Dense(90, activation='softmax'))

        self.single_d_model.compile(loss='categorical_crossentropy',
                                    optimizer=keras.optimizers.Adam(lr=learning_rate, epsilon=None, decay=0.0),
                                    metrics=['accuracy'])

        self.single_d_history = self.single_d_model.fit(x_train, y_train,
                                                        batch_size=batch_size,
                                                        epochs=epochs,
                                                        verbose=1, validation_split=0.1)

        save1 = input('Save? y/n')
        if save1 == 'y':
            self.single_d_model.save('MODELS/1_LAYER_D_NET.h5')

    def single_sld(self, x_train, y_train, batch_size, epochs, learning_rate, l2_lambda):
        y_train = y_train[:, 0]
        y_train = np.round(y_train, 2)
        y_train_dict = dict(zip(np.unique(y_train), range(len(np.unique(y_train)))))
        y_train = np.vectorize(y_train_dict.get)(y_train)
        y_train = to_categorical(y_train)

        inputshape = (None, 2, 1)

        self.single_sld_model = Sequential()

        self.single_sld_model.add(
            Conv2D(input_shape=inputshape, filters=64, kernel_size=(3, 2), strides=(1, 1), padding='same',
                   kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.single_sld_model.add(BatchNormalization())
        self.single_sld_model.add(Activation('relu'))
        self.single_sld_model.add(MaxPooling2D(pool_size=(3, 1), padding='same'))

        self.single_sld_model.add(Conv2D(filters=32, kernel_size=(3, 2), strides=(1, 1), padding='same'
                                         , kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.single_sld_model.add(BatchNormalization())
        self.single_sld_model.add(Activation('relu'))
        self.single_sld_model.add(MaxPooling2D(pool_size=(3, 1), padding='same'))

        self.single_sld_model.add(Conv2D(filters=16, kernel_size=(3, 2), strides=(1, 1), padding='same'
                                         , kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.single_sld_model.add(BatchNormalization())
        self.single_sld_model.add(Activation('relu'))
        self.single_sld_model.add(AveragePooling2D(pool_size=(3, 1), padding='same'))

        self.single_sld_model.add(Dense(256, kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.single_sld_model.add(Activation('relu'))
        self.single_sld_model.add(BatchNormalization())
        self.single_sld_model.add(Dropout(0.2))
        self.single_sld_model.add(Dense(512, kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.single_sld_model.add(Activation('relu'))
        self.single_sld_model.add(BatchNormalization())
        self.single_sld_model.add(Dropout(0.2))
        self.single_sld_model.add(Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(l2_lambda)))
        self.single_sld_model.add(Activation('relu'))
        self.single_sld_model.add(BatchNormalization())
        self.single_sld_model.add(Dropout(0.2))
        self.single_sld_model.add(GlobalMaxPooling2D())
        self.single_sld_model.add(Dense(105, activation='softmax'))

        self.single_sld_model.compile(loss='categorical_crossentropy',
                                      optimizer=keras.optimizers.Adam(lr=learning_rate, epsilon=None, decay=0.0),
                                      metrics=['accuracy'])

        self.single_sld_history = self.single_sld_model.fit(x_train, y_train,
                                                            batch_size=batch_size,
                                                            epochs=epochs,
                                                            verbose=1, validation_split=0.1, shuffle=True)
        save = input('Save? y/n')
        if save == 'y':
            self.single_sld_model.save('MODELS/1_LAYER_SLD_NET.h5')

    def load_model(self, net_name, path):
        if net_name == 'n_class_model':
            self.n_class_model = keras.models.load_model(path)
        elif net_name == 'single_d_model':
            self.single_d_model = keras.models.load_model(path)
        elif net_name == 'single_sld_model':
            self.single_sld_model = keras.models.load_model(path)
        elif net_name == 'multi_d_model':
            self.multi_d_model = keras.models.load_model(path)
        elif net_name == 'multi_sld_model':
            self.multi_sld_model = keras.models.load_model(path)

    def peak_detection(self, output_list, n_highest):
        peak_list = []
        spikes = []
        for n in range(len(output_list)):
            if n == 0 and output_list[0][0] > output_list[1][0]:
                peak_list.append(output_list[n])
            elif n == len(output_list) - 1 and output_list[n][0] > output_list[n - 1][0]:
                peak_list.append(output_list[n])
            elif output_list[n][0] > output_list[n - 1][0] and output_list[n][0] > output_list[n + 1][0]:
                peak_list.append(output_list[n])
        peak_list = sorted(peak_list, key=lambda x: x[0])[n_highest:]
        # if len(peak_list) >= 2:
        #     peak_list = sorted(peak_list, key=lambda x: x[0])
        # else:
        #     peak_list = sorted(output_list, key=lambda x: x[0])
        # for i in range(n_highest):
        #     spikes.append(peak_list[-(i+1)])

        return peak_list #spikes

    def result_parsing(self, nn_output, sld_or_d, layer_num):
        '''_1 suffix denotes output from 1st branch of network, _2 denotes output from second'''
        if layer_num ==1:
            if sld_or_d == 'sld':
                out_list_1 = list(zip(nn_output[0], self.sld_bins))
            elif sld_or_d == 'd':
                out_list_1 = list(zip(nn_output[0], self.d_bins))
            spikes_1 = self.peak_detection(out_list_1, 4)
            value_confidence_1 = np.sum(spikes_1, axis=0)[0]

            return [spikes_1, value_confidence_1]

        if layer_num == 2:
            if sld_or_d == 'sld':
                out_list_1 = list(zip(nn_output[0][0], self.sld_bins))
                out_list_2 = list(zip(nn_output[1][0], self.sld_bins))
            elif sld_or_d == 'd':
                out_list_1 = list(zip(list(nn_output[0][0]), self.d_bins))
                out_list_2 = list(zip(list(nn_output[1][0]), self.d_bins))

            ### peak detection ###
            spikes_1 = self.peak_detection(out_list_1, 4)
            spikes_2 = self.peak_detection(out_list_2, 4)


            ### confidence/probability metrics #
            value_confidence_1 = np.sum(spikes_1, axis=0)[0]
            value_confidence_2 = np.sum(spikes_2, axis=0)[0]

            #######################################

            return [spikes_1, value_confidence_1], [spikes_2, value_confidence_2]

    def analyse(self, x):
        layer_num = np.argmax(self.n_class_model.predict(x))

        if layer_num == 0:
            self.sample1 = []
        if layer_num == 1:
            d = self.single_d_model.predict(x)
            d = categorical_to_d(d)
            sld = self.single_sld_model.predict(x)
            sld = categorical_to_sld(sld)
            self.sample1.append(Layer(d, sld))
        if layer_num == 2:
            d1_list = self.multi_d_model.predict(x)[0]
            d1_list = list.zip(d1_list, self.d_bins)
            d1_list = np.sort(d1_list, axis=1)
            d1 = d1_list[-1]
            d2 = d2_list[-2]
            # d2_list = self.multi_d_model.predict(x)[1]

            sld1_list = self.multi_sld_model.predict(x)[0]
            sld1_list = list.zip(sld1_list, self.sld_bins)
            sld1_list = np.sort(sld1_list, axis=1)
            sld1 = sld1_list[-1]
            sld2 = sdl1_list[-2]
            # sld2 = self.multi_sld_model.predict(x)[1]

            self.sample1.append(Layer(d1, sld1))
            self.sample1.append(Layer(d2, sld2))
# def testfunc():
#     test = nn()
#     test.load_model('multi_sld_model', 'MODELS\\2_LAYER_SLD_NET.h5')
#     xtest = np.loadtxt('datatester.csv', delimiter=',')[:,:2]
#     xtest = xtest.reshape((1,)+xtest.shape)
#     xtest = preprocess_x(xtest)
#     ree = test.multi_sld_model.predict(xtest)
#     res = test.result_parsing(ree, 'sld', 2)
#     print(res[0])
#
# testfunc()
import os
os.environ["PATH"] += os.pathsep + 'C:\\Users\\Jonathan Xue\\Anaconda3\\envs\\Lib\\site-packages\\Graphviz2.38\\bin'
from keras.utils.vis_utils import plot_model
net = nn()
#net.load_model('multi_sld_model', 'MODELS\\2_LAYER_SLD_NET.h5')
net.load_model('multi_d_model', 'MODELS\\2_LAYER_D_NET.h5')
#plot_model(net.multi_sld_model, to_file='multi_sld_model.png')
plot_model(net.multi_d_model, to_file='multi_d_model.png')
#print(net.multi_sld_model.summary())