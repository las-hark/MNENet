import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.constraints import max_norm
from keras.layers import Conv2D, BatchNormalization, AveragePooling2D, Flatten, Dense, Dropout, Activation
from keras.layers import DepthwiseConv2D, SeparableConv2D
from keras import backend as K
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch,BayesianOptimization
# data process functions

def prepro_x(x: np.array, downrate : int) -> np.array:
    """down sampling from different time points to augmentate data"""

    x_agu = np.zeros((downrate*x.shape[0], x.shape[1], int(x.shape[2]/downrate)))
    for k in range(len(x)):
        for i in range(downrate):
            x_agu[downrate*k + i] = x[k, :, i::downrate]
    return x_agu


def getdata(downrate : int) -> np.array:
    """read data, downsample, split train set into train set and validation set"""

    x_train_ori = np.load(r'cross_train_set_nor.npy')
    x_test1_ori = np.load(r'cross_test1_set_nor.npy')
    x_test2_ori = np.load(r'cross_test2_set_nor.npy')
    x_test3_ori = np.load(r'cross_test3_set_nor.npy')

    x_train_agu = np.array(prepro_x(x_train_ori, downrate), dtype=np.float32)
    x_test1_agu  = np.array(prepro_x(x_test1_ori, downrate), dtype=np.float32)
    x_test2_agu  = np.array(prepro_x(x_test2_ori, downrate), dtype=np.float32)
    x_test3_agu  = np.array(prepro_x(x_test3_ori, downrate), dtype=np.float32)

    y_train0 = np.eye(4)[np.repeat(np.load(r'cross_train_label.npy') - 1, downrate)]
    y_test1  = np.eye(4)[np.repeat(np.load(r'cross_test1_label.npy') - 1, downrate)]
    y_test2  = np.eye(4)[np.repeat(np.load(r'cross_test2_label.npy') - 1, downrate)]
    y_test3  = np.eye(4)[np.repeat(np.load(r'cross_test3_label.npy') - 1, downrate)]

    # reshape: numbers, channels, samplingrate, 1
    x_train0 = x_train_agu.reshape(x_train_agu.shape[0], x_train_agu.shape[1], x_train_agu.shape[2], 1)
    x_test1   = x_test1_agu.reshape(x_test1_agu.shape[0], x_test1_agu.shape[1], x_test1_agu.shape[2], 1)
    x_test2   = x_test2_agu.reshape(x_test2_agu.shape[0], x_test2_agu.shape[1], x_test2_agu.shape[2], 1)
    x_test3   = x_test3_agu.reshape(x_test3_agu.shape[0], x_test3_agu.shape[1], x_test3_agu.shape[2], 1)
    
    x_train, x_val, y_train, y_val = train_test_split(x_train0, y_train0, test_size=0.2, random_state=42)
    
    return x_train, y_train, x_val, y_val, x_test1, y_test1, x_test2, y_test2, x_test3, y_test3


def myGenerator(x : np.array, y : np.array, batch_size : int):
    """set generator for model fitting to load data piece by piece
        OOM can be avoided in this way"""

    total_size = len(x)
    while 1:
        for i in range(total_size // batch_size):
            yield x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]
    return myGenerator


def build_model(hp):
    model = Sequential()

    nb_conduction = input_shape[0]
    FZ = 2034/downrate

    # Block 1
    model.add(Conv2D(
        filters=4,
        kernel_size=(1, int(FZ/2)),
        padding='same',
        input_shape=input_shape,
        kernel_regularizer=keras.regularizers.l1(hp.Choice('l1', values = [0.01,0.001]),
        
        use_bias=False
    ))
    model.add(BatchNormalization(axis=-1))
    model.add(DepthwiseConv2D(
        kernel_size=(nb_conduction, 1),
        padding='valid',
        depthwise_constraint=max_norm(1.),
        depth_multiplier=2,
        use_bias=False
    ))
    model.add(Dropout(hp.Choice('dropout_1', values = [0.25,0.5,0.75,0.9])))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('elu'))
    model.add(AveragePooling2D(pool_size=(1, 4)))

    # Block 2
    model.add(SeparableConv2D(
        filters=4,
        kernel_size=(1, int(FZ/4)),
        padding='same',
        use_bias=False
    ))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('elu'))
    model.add(AveragePooling2D(pool_size=(1, 8)))
    model.add(Dropout(hp.Choice('dropout_2', values = [0.25,0.5,0.75,0.9])))

    # Block 3
    model.add(Flatten())
    model.add(Dense(
        units=nb_classes,
        activation='softmax',
        kernel_constraint=max_norm(0.25)
    ))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate =(hp.Choice('lr', values = [0.01,0.001]))),
        metrics=['accuracy']
    )

    return model

def get_tuner():
    return BayesianOptimization(
        build_model,
        objective='val_loss',
        max_trials=20,
        directory='tuner_dir',
        project_name='eegnet_tuning',
        overwrite=True
    )


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(tf.config.list_physical_devices('GPU'))
    tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # programs can only use up to 50% of a given gpu memory
    config.gpu_options.allow_growth = False  
    sess = tf.compat.v1.Session(config = config)

    downrate = 8
    x_train, y_train, x_val, y_val, x_test1, y_test1, x_test2, y_test2, x_test3, y_test3 = getdata(downrate)
    nb_classes = 4
    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    tuner = get_tuner()

    tuner.search(myGenerator(x_train, y_train, batch_size = 8), steps_per_epoch=len(x_train) // 8, epochs = 20, validation_data=(x_val, y_val))
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps.values)
    # tuner.results_summary()
    # Build the model with the best hyperparameters and train it
    best_model = build_model(best_hps)
    a = 0
