from localreg import RBFnet, plot_corr
from localreg.metrics import rms_error, rms_rel_error
import matplotlib.pyplot as plt

def rbf_network(Is,Ts,M):

    # Train by minimizing relative error in temperature
    net = RBFnet()
    net.train(Is[:M], Ts[:M], num=200, relative=True, measure=rms_rel_error) ### 180/200 works okay/250

    # Plot and print error metrics on test data
    pred = net.predict(Is[M:])

    #Vs_geo1_str=np.array2string(Vs_geo1, formatter={'float_kind':lambda x: "%.1f" % x})
    #Vs_geo2_str=np.array2string(Vs_geo2, formatter={'float_kind':lambda x: "%.1f" % x})
    fig, ax = plt.subplots()
    plot_corr(ax, Ts[M:], pred,log=True)
    plt.title('NN correlation plot')
    plt.savefig('correlation.png', bbox_inches="tight")
    plt.show()
    return pred,net
