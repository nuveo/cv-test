# Create a conda env to run DocumentCleanup

This work was developed and tested in a system with a processor AMD Ryzen 1800X, 16 GBs of RAM memory and a NVIDIA GTX 1080 ti graphics processor, running a Ubuntu LTS 16.04 and NVIDIA Version was  version 10.1 

```
$> conda create -n nuveo python=3.6
```
```
$> conda install -c conda-forge opencv
```
```
$> conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
```
$> conda install -c anaconda jupyter scipy matplotlib
```
```
$> pip install trdg
```
# Run

The notebook **apply_model.ipynb** is used to apply my model to a directory with text images to be cleanned.  

And use the notebook **create_model.ipynb** if you want to recriate and train my proposal model to cleanup documents.
