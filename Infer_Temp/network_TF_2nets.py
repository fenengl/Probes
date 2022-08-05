import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from localreg import RBFnet, plot_corr


def tensorflow_network(Is_sph,Is_cyl,Ts,M):

        # ## normalizer
    normal1 = preprocessing.Normalization()
    normal1.adapt(Is_sph[:M])
    input_size1 = 1
    print(input_size1)

    normal2 = preprocessing.Normalization()
    normal2.adapt(Is_cyl[:M])
    input_size2 = Is_cyl[:M].shape[1]
    print(input_size2)
    import tensorflow as tf

    #layers. Dense-- regular NN layer densely connected
    A = keras.Sequential([
    normal1,
    layers.Dense(40, activation='relu', input_shape=[1]), #50
    layers.Dense(20, activation='relu',),
    layers.Dense(1, activation='relu')
    ],
    name="A")

    #tf_model1.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001))#,metrics=['accuracy']) #remove acc ## works also with logarithmic, when increasing learning rate, finishes in less steps but less accurate


    B = keras.Sequential([
    normal2,
    layers.Dense(40, activation='relu', input_shape=[1]), #50
    layers.Dense(40, activation='relu',),
    layers.Dense(4, activation='relu')
    ],
    name="B")

    #     # define two sets of inputs
    # inputA = Input(shape=(32,))
    # inputB = Input(shape=(128,))
    # # the first branch operates on the first input
    # x = Dense(8, activation="relu")(inputA)
    # x = Dense(4, activation="relu")(x)
    # x = Model(inputs=inputA, outputs=x)
    # # the second branch opreates on the second input
    # y = Dense(64, activation="relu")(inputB)
    # y = Dense(32, activation="relu")(y)
    # y = Dense(4, activation="relu")(y)
    # y = Model(inputs=inputB, outputs=y)
    # combine the output of the two branches
    combined = layers.concatenate([A.output, B.output])
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = layers.Dense(5, activation="relu")(combined)
    z = layers.Dense(1, activation="relu")(z)
    # our model will accept the inputs of the two branches and
    # then output a single value
    merged_model = keras.Model(inputs=[A.input, B.input], outputs=z)


    ## relu, swish,elu,selu,gelu
    #tf_model2.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001))#,metrics=['accuracy']) #remove acc ## works also with logarithmic, when increasing learning rate, finishes in less steps but less accurate
    # merged_input = tf.keras.Input((1,))
    # x = A(merged_input)
    # merged_output = B(x)
    # merged_model = tf.keras.Model(inputs=merged_input, outputs=merged_output, name="merged_AB")
    #
    merged_model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001))#,metrics=['accuracy'])
    merged_model.summary()
    #
    #
    # a_b = concatenate([al_4,bl_2],name="concatenated_layer")

#Final Layer
# output_layer = Dense(16, activation = "sigmoid", name = "output_layer")(a_b)



    with open('report.txt','w') as fh:
        merged_model.summary(print_fn=lambda x: fh.write(x + '\n'))

        history=merged_model.fit([Is_sph[:M],Is_cyl[:M]], Ts[:M],
               validation_data=([Is_sph[M:],Is_cyl[M:]], Ts[M:]), ## add this again
               verbose=1, epochs=60)

        results = merged_model.evaluate([Is_sph[M:],Is_cyl[M:]], Ts[M:], verbose=1)

        pred=merged_model.predict([Is_sph[M:],Is_cyl[M:]])


        fig, ax = plt.subplots()
        plot_corr(ax, Ts[M:], pred,log=True)
        plt.title('NN correlation plot tf')
        plt.savefig('correlation.png', bbox_inches="tight")
        plt.show()

        return pred,results,history,merged_model










        #
        # # Image
        # input_1 = tf.keras.layers.Input(shape=[1])
        # dense1_1 = tf.keras.layers.Dense(40, activation=tf.keras.activations.relu)(input_1)
        #
        # # Second conv layer :
        # dense1_2 = tf.keras.layers.Dense(40, activation=tf.keras.activations.relu)(dense1_1)
        # dense1_3 = tf.keras.layers.Dense(1, activation=tf.keras.activations.relu)(dense1_2)
        # # Flatten layer :
        # flatten = tf.keras.layers.Flatten()(dense1_3)
        #
        # # The other input
        # input_2 = tf.keras.layers.Input(shape=[2])
        # dense2_1 = tf.keras.layers.Dense(40, activation=tf.keras.activations.relu)(input_2)
        # dense2_2 = tf.keras.layers.Dense(40, activation=tf.keras.activations.relu)(dense2_1)
        # dense2_3 = tf.keras.layers.Dense(1, activation=tf.keras.activations.relu)(dense2_2)
        #
        # # Concatenate
        # concat = tf.keras.layers.Concatenate()([flatten, dense2_3])
        #
        # n_classes = 4
        # # output layer
        # output = tf.keras.layers.Dense(units=n_classes,
        #                                activation=tf.keras.activations.softmax)(concat)
        #
        # full_model = tf.keras.Model(inputs=[input_1, input_2], outputs=[output])
        # full_model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(0.001))#,metrics=['accuracy'])
        #
        # print(full_model.summary())
        #
        #
        #
        # train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
        #
        # model.fit(train_x, train_y, epochs=5, batch_size=1, verbose=1)
        #
        #
        # with open('report.txt','w') as fh:
        #     tf_model.summary(print_fn=lambda x: fh.write(x + '\n'))
        #
        #     history=tf_model.fit(Is[:M], Ts[:M],
        #            validation_data=(Is[M:], Ts[M:]), ## add this again
        #            verbose=1, epochs=60)
        #
        #     results = tf_model.evaluate(Is[M:], Ts[M:], verbose=1)
        #
        #     pred=tf_model.predict(Is[M:])
        #
        #
        #     fig, ax = plt.subplots()
        #     plot_corr(ax, Ts[M:], pred,log=True)
        #     plt.title('NN correlation plot tf')
        #     plt.savefig('correlation.png', bbox_inches="tight")
        #     plt.show()
        #
        #     return pred,results,history,tf_model
