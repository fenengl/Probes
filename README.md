
<!-- This repository consists of three folders.
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

In PTetraProbes, the simulations runs on PTetra are specified. -->

# Probes
A python tool to infer temperature for different Langmuir probe set-ups.

## Setting up Workspace
Make a clone of the `Probes` repository using the following commands,
```shell
git clone https://github.com/fenengl/Probes.git
```
Go to the root folder,
```shell
cd Probes
```
### Installing with Anaconda

An [Anaconda](www.anaconda.com) environment is available. Before following the rest of the instruction, make sure Anaconda is installed. You can create the `probes` environment with the following command:
```shell
conda env create
```
To activate the environment, run:
```shell
conda activate probes
```
and to deactivate it, run:
```shell
conda deactivate
```
## Getting started
Go to the source folder,
```shell
cd probes
```
To get all the options, use the following command,
```shell
python main.py -h
```
There are six used cases for the model. Three for each geometry types (`Cylinder` and `Sphere`). The used cases can be called if the user intends not to generate synthetic data.
### Examples
 `Case - I`: 3 Cylinder ()
 ```shell
 python main.py -p cylinder -v 1
 ```
 `Case - II`: 3 Cylinder ()
```shell
 python main.py -p cylinder -v 2
 ```
 `Case - III`: 3 Cylinder ()
 ```shell
 python main.py -p cylinder -v 3
 ```
 `Case - IV`: 3 Sphere ()
 ```shell
 python main.py -p sphere -v 1
 ```
 `Case - V`: 3 Sphere ()
  ```shell
 python main.py -p sphere -v 2
 ```
 `Case - VI`: 3 Sphere ()
  ```shell
 python main.py -p sphere -v 3
 ```
Note: If you are using it for the first time, you should train your network irrespective of the data. In such case use additional argument `-t`.
e.g.
 ```shell
 python main.py -p cylinder -v 1 -t
 ```
### Figures
Figures corresponding to the cases can be found under the directory: `probes/images`
