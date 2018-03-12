# Blob Segmentation with U-Net

@author: Ali Hashmi

@note: The python script uses MXNET (a modern opensource deep learning framework) to implement U-net: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net

The neural network is general purpose but the script is not. Therefore, the script needs to be customized for other segmentation 
based applications (requiring the search for  new hyperparameters)

The convolutional neural network was sufficiently trained - the training-dataset consists of images and labels - over an Nvidia GTX 1050 GPU with 640 CUDA cores. The training takes a matter of minutes. Training over a CPU is possible in case a GPU is not available, but it will take considerably longer time (usually several hours due to a lower number of processor cores).

The network once trained can be applied on an image to generate the segmentation mask

### Example:

#### Original Image

![alt text](https://github.com/alihashmiii/blobsegmentation/blob/master/for%20readme/image300.png)




#### Ground Truth (from human)

![alt text](https://github.com/alihashmiii/blobsegmentation/blob/master/for%20readme/Mask300.png)



#### Segmentation Output (from U-Net)

![alt text](https://github.com/alihashmiii/blobsegmentation/blob/master/for%20readme/segmentationOutput.png)



###### There is a lot of room for improvement. The gaps in the segmentation result can be closed with some morphological image processing operations such as a Dilation/Closing operation. Perhaps the results can be improved with sufficient training and suitable choice of hyperparameters.

To do:
- improve the segmentation results
- resize output back ?
- implement the network in Mathematica
- find better hyperparameters and evaluation_metric
