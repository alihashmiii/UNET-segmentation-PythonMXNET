# Blob Segmentation with U-Net

@author: Ali Hashmi

@note: The python script uses MXNET (a modern opensource deep learning framework) to implement U-net: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net

The neural network is general purpose but the script is not. Therefore, the script needs to be customized for other segmentation 
based applications (requiring the search for  new hyperparameters)

The convolutional neural network is trained sufficiently with a training-dataset (consists of images and labels). The network once trained can be applied on an image to generate the segmentation mask

Example:

#### Original Image:

![alt text](https://github.com/alihashmiii/blobsegmentation/blob/master/for%20readme/image300.tif)




#### Ground Truth:

![alt text](https://github.com/alihashmiii/blobsegmentation/blob/master/for%20readme/Mask300.tif)



#### Segmentation Output:

![alt text](https://github.com/alihashmiii/blobsegmentation/blob/master/for%20readme/segmentationOutput.tif)



"the gaps can be closed with Morphological Image processing operations such as Closing. Perhaps the results can be improved with sufficient training and suitable choice of hyperparameters"
