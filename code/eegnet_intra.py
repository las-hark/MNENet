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

# data process functions

def prepro_x(x: np.array, downrate : int) -> np.array:
    """down sampling from different time points to augmentate data"""

    x_agu = np.zeros((downrate*x.shape[0], x.shape[1], int(x.shape[2]/downrate)))
    for k in range(len(x)):
        for i in range(downrate):
            x_agu[downrate*k + i] = x[k, :, i::downrate]
    return x_agu

def getdata(downrate : int):
    """read data, downsample, split train set into train set and validation set"""

    x_train_ori = np.load(r'intra_train_set_nor.npy')
    x_test_ori  = np.load(r'intra_test_set_nor.npy')

    x_train_agu = np.array(prepro_x(x_train_ori, downrate), dtype=np.float32)
    x_test_agu  = np.array(prepro_x(x_test_ori, downrate), dtype=np.float32)

    y_test   = np.eye(4)[np.repeat(np.load(r'intra_test_label.npy') - 1, downrate)]
    y_train0 = np.eye(4)[np.repeat(np.load(r'intra_train_label.npy') - 1, downrate)]

    # reshape: numbers, channels, samplingrate, 1
    x_train0 = x_train_agu.reshape(x_train_agu.shape[0], x_train_agu.shape[1], x_train_agu.shape[2], 1)
    x_test   = x_test_agu.reshape(x_test_agu.shape[0], x_test_agu.shape[1], x_test_agu.shape[2], 1)
    
    x_train, x_val, y_train, y_val = train_test_split(x_train0, y_train0, test_size=0.2, random_state=42)
    
    return x_train, y_train, x_val, y_val, x_test, y_test

# model functions

def EEGNet(input_shape : tuple, nb_classes : int, fre_sampling: float):
    """build EEGNET"""

    model = Sequential()
    nb_conduction = input_shape[0]  
    FZ = fre_sampling
    # block 1 
    # 2dconv and depthwiseconv: extract time and spatial feature
    model.add(Conv2D(filters = 4, 
                    kernel_size = (1, int(FZ/2)),
                    padding = 'same',
                    input_shape = input_shape,
                    kernel_regularizer=keras.regularizers.l1(0.01),
                    use_bias = False))
    model.add(BatchNormalization(axis = -1))
    model.add(DepthwiseConv2D(kernel_size = (nb_conduction, 1), 
                            padding = 'valid',
                            depthwise_constraint = max_norm(1.),
                            depth_multiplier = 2,
                            use_bias = False))
    model.add(Dropout(0.5))
    model.add(BatchNormalization(axis = -1))
    model.add(Activation('elu'))
    model.add(AveragePooling2D(pool_size = (1, 4)))


    # block 2 
    # separableconv : extract feature on different time scales
    model.add(SeparableConv2D(filters = 4, 
                            kernel_size = (1, int(FZ/4)),
                            padding = 'same',
                            use_bias = False))
    model.add(BatchNormalization(axis = -1))
    model.add(Activation('elu'))
    model.add(AveragePooling2D(pool_size = (1, 8)))
    model.add(Dropout(0.25))

    # block 3 
    # flatten and fully connected layers: classify
    model.add(Flatten())
    model.add(Dense(nb_classes, 
            activation = 'softmax',
            kernel_constraint = max_norm(0.25)))

    return model


def myGenerator(x : np.array, y : np.array, batch_size : int):
    """set generator for model fitting to load data piece by piece
        OOM can be avoided in this way"""

    total_size = len(x)
    while 1:
        for i in range(total_size // batch_size):
            yield x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]
    return myGenerator


def train_eegnet(X : np.array, Y : np.array, X_val : np.array, Y_val : np.array, batch_size : int, nb_epoch : int) :
    """Train model, save model, plot training history, save training history"""

    # !!!!!!!!change the path every run!!!!!!!!!! better use a breakpoint here
    model_savepath = r'testmodel9.h5'    
    trainloss_savepath = r'trainloss_tmd9.npy'
    validloss_savepath = r'validloss_tmd9.npy'
    checkpoint_path = r'bestmodel.h5'
    # Training  
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
                                                monitor='val_loss',
                                                verbose = 1,
                                                save_best_only=True,
                                                save_freq= 'epoch') 
    eegnet_model.compile(loss = 'categorical_crossentropy',
                        optimizer = 'adam')
                        #metrics = ['categorical_accuracy']
    
    history = eegnet_model.fit_generator(generator = myGenerator(X, Y, batch_size),
                                        steps_per_epoch = len(X)//(batch_size), 
                                        validation_data = (X_val, Y_val),                            
                                        epochs     = nb_epoch, 
                                        verbose    = 1,
                                        callbacks  = [checkpoint])                

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    # plot the training and validation loss
    plt.plot(epochs, train_loss, '-', color = 'red',label='Training loss')
    plt.plot(epochs, val_loss, '--', color ='green',label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # save the final trained model and history
    eegnet_model.save(model_savepath)
    np.save(trainloss_savepath, train_loss)
    np.save(validloss_savepath, val_loss)

    return history, model_savepath


def predict_eegnet(model_loadpath):
    """predict"""

    model = keras.models.load_model(model_loadpath)
    predictions = np.zeros((len(y_test)))

    for i in range(len(x_test)):
        probability = model.predict(x_test[i:i+1])
        predictions[i] = probability.argmax(axis = -1)
    
    return predictions



if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(tf.config.list_physical_devices('GPU'))
    tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # programs can only use up to 50% of a given gpu memory
    config.gpu_options.allow_growth = False  
    sess = tf.compat.v1.Session(config = config)

    downrate = 8 #downsampling rate
    x_train, y_train, x_val, y_val, x_test, y_test = getdata(downrate)
    input_shape = (x_train.shape[1], x_train.shape[2], 1) #inputshape = (channels, time, 1) e.i.(248, 35624, 1)


    # # built model
    nb_classes = 4  #classification numbers
    eegnet_model = EEGNet(input_shape, nb_classes, 2034/downrate)
    # print model
    eegnet_model.summary()
    # train model
    history, modelpath = train_eegnet(x_train, y_train, x_val, y_val, batch_size = 8, nb_epoch = 100)
    # predict
    y_pre = predict_eegnet(modelpath)
    # label from one-hot to normal
    label = np.array([one_label.tolist().index(1) for one_label in y_test], dtype = np.float64)
    pre_accuracy = np.sum(label == y_pre) / len(y_pre)
    print(y_pre)
    print(pre_accuracy)
