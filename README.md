<<<<<<< HEAD
This repository consists of three folders.
Infer_Temp
extrapolation_verification
PTetraProbes

In Infer_Temp the temperature for different Langmuir probe set-ups can be inferred. The probe set ups are in sub-folders. The data is generated with data_gen.py
The versions of the 3 cylinder setup are e.g. called main_3cyl_a.py,  main_3cyl_b.py,  main_3cyl_c.py
They are to infer temperature.
main_3cyl_ne.py is to infer the electron density.
main_3cyl_Vb.py is to infer the floating potential.
The same goes for the 3 spheres and the mNLP folder.

To run the code one can run the main files.  The tensorflow models are saved and used for the run to reproduce the exact figures of the publication. The flags to generate data or train a new tensorflow model can be set from False to True and new data and model will be generated.

The files network_TF_DNN.py is the machine learning network used for Te and Vb inference.
The network_TF_ne.py used for electron temperature.

The noise folder within Infer_Temp shows the different cases with added noise.

In the extrapolation verification folder, Figure 1 of the publication is plotted.

In PTetraProbes, the simulations runs on PTetra are specified.
=======

>>>>>>> d7cfe66b005a159708e9d97331273e07411f5968
