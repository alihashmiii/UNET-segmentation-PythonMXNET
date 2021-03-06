# UNet Segmentation Python/MXNET 

![status](https://img.shields.io/badge/status-passed-blue.svg)

`@author`: Ali Hashmi

`@note`: The python script uses `MXNET` (a modern opensource deep learning framework) to implement U-net: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net. You may need to install all the appropriate libraries/modules for python: `mxnet` (configure it for GPU), `cv2` (opencv), `numpy`, `matplotlib` etc.. As of now, the script has only been tested on Windows.
(1) Download the script files to the same folder (2) Provide the folder path to `sys.path` or add the folder path to PYTHONPATH in the environment variables of your system.

The neural network is general purpose. Please modify and/or customize the scripts for your segmentation problem (possibly it requires the search for new hyperparameters)

The convolutional neural network (`CNN`) was trained sufficiently - the training-dataset consists of images and labels - over an Nvidia GTX 1050 GPU (640 CUDA cores). The training takes a matter of a few minutes. Training over CPU is possible in case a GPU is not available, albeit it will take a considerably longer period (usually several hours due to a lower number of processor cores).

The neural-network once trained can be applied on an image to generate the segmentation mask

### Example:

![alt-text](https://github.com/alihashmiii/UNET-segmentation-PythonMXNET/blob/master/for%20readme/accuracyNet.png)


A filling transform was done to fill in the defect (hole) in the mask before it was overlayed on the image (using Mathematica).


### Accuracy check for a batch of unseen images (97.69 %)

`@Note`: the Mathematica/Wolfram Language version of the trained neural network (https://github.com/alihashmiii/UNet-Segmentation-Wolfram) is doing considerably better than the Python MXNET version, despite the fact that Mathematica uses MXNET at the backend for implementing neural nets. This is perhaps due to the thresholding operation that i need to apply after the logisticRegression layer. Nevertheless, it is doing a reasonable job segmenting most aggregates (with or without applying any additional morphological operations)

![alt text](https://github.com/alihashmiii/blobsegmentation/blob/master/for%20readme/accuracy.png)


##### Output from UNet after thresholding, filling and removing small components

![alt-text](https://github.com/alihashmiii/UNET-segmentation-PythonMXNET/blob/master/for%20readme/outputUnet.png)


`To do`: 
- resize output back using perhaps bilinear interpolation
