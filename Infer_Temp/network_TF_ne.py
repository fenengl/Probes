import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from localreg import RBFnet, plot_corr


def tensorflow_network(par_in,var,M):
    print(par_in[:M])
    ## normalizer
    normal = preprocessing.Normalization()
    normal.adapt(par_in[:M])
    input_size = par_in[:M].shape[1]

    #layers. Dense-- regular NN layer densely connected
    tf_model = keras.Sequential([
    normal,
    layers.Dense(100, activation='relu'),
    layers.Dense(40, activation='selu'),
    layers.Dense(40, activation='gelu'),
    layers.Dense(40, activation='selu'),
    layers.Dense(40, activation='relu'),
    layers.Dense(40, activation='selu'),
    layers.Dense(40, activation='gelu'),
    layers.Dense(40, activation='selu'),
    layers.Dense(40, activation='relu'),
    layers.Dense(1)
    ])
    ## relu, swish,elu,selu,gelu

    tf_model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001))#,metrics=['accuracy']) #remove acc ## works also with logarithmic, when increasing learning rate, finishes in less steps but less accurate

    with open('report.txt','w') as fh:
        tf_model.summary(print_fn=lambda x: fh.write(x + '\n'))

        history=tf_model.fit(par_in[:M], var[:M],
               validation_data=(par_in[M:], var[M:]), ## add this again
               verbose=1, epochs=60)

        results = tf_model.evaluate(par_in[M:], var[M:], verbose=1)

        pred=tf_model.predict(par_in[M:])


        fig, ax = plt.subplots()
        plot_corr(ax, var[M:], pred,log=True)
        plt.title('NN correlation plot tf')
        plt.savefig('correlation.png', bbox_inches="tight")
        plt.show()

        return pred,results,history,tf_model
