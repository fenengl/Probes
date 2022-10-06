import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from localreg import RBFnet, plot_corr
import matplotlib as mplot

def tensorflow_network(Is,Ts,M,K):

    ## normalizer
    normal = preprocessing.Normalization()
    normal.adapt(Is[:M])

    #layers. Dense-- regular NN layer densely connected
    tf_model = keras.Sequential([
    normal,
    layers.Dense(40, activation='relu', input_shape=(Is[:M].shape[1],)),
    #layers.Dropout(0.2),#50
    #layers.Dropout(0.1),
    layers.Dense(40, activation='relu'),

    layers.Dense(1, activation='relu'),

    ])
    ## relu, swish,elu,selu,gelu
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    tf_model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001),metrics=['MeanAbsolutePercentageError']) #remove acc ## works also with logarithmic, when increasing learning rate, finishes in less steps but less accurate

    with open('report.txt','w') as fh:
        tf_model.summary(print_fn=lambda x: fh.write(x + '\n'))

    history=tf_model.fit(Is[:M], Ts[:M],
        validation_data=(Is[M:K], Ts[M:K]),
        callbacks=[callback], ## add this again
        verbose=1, epochs=150)

    results = tf_model.evaluate(Is[M:K], Ts[M:K], verbose=1)

    return results,history,tf_model
