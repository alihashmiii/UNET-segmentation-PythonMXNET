# Blob Segmentation with U-Net

`Status`: passed

`@author`: Ali Hashmi

`@note`: The python script uses `MXNET` (a modern opensource deep learning framework) to implement U-net: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net

The neural network is general purpose. Please modify and/or customize the scripts for your segmentation problem (possibly it requires the search for new hyperparameters)

The convolutional neural network (`CNN`) was trained sufficiently - the training-dataset consists of images and labels - over an Nvidia GTX 1050 GPU (640 CUDA cores). The training takes a matter of a few minutes. Training over CPU is possible in case a GPU is not available, albeit it will take a considerably longer period (usually several hours due to a lower number of processor cores).

The neural-network once trained can be applied on an image to generate the segmentation mask

### Example:

#### Original Image

![alt text](https://github.com/alihashmiii/blobsegmentation/blob/master/for%20readme/image300.png)




#### Ground Truth (from human)

![alt text](https://github.com/alihashmiii/blobsegmentation/blob/master/for%20readme/Mask300.png)



#### Segmentation Output (from U-Net)

![alt text](https://github.com/alihashmiii/blobsegmentation/blob/master/for%20readme/segmentationOutput.png)




`To do`: 
- resize output back 
