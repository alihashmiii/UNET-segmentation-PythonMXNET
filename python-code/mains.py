# U-ConvNet for segmentation of aggregates
print("""The script is a Python implementation of UConvNet for image segmentation. We provide a labeled training dataset
- images and corresponding masks - to train the neural net. The script makes use of MXNET library/APIs, functions from
OpenCV (CLAHE) and the python libraries:
numpy, matplotlib, os and PIL\n
/**
 * @author :Ali Hashmi (with input from Alexey Golyshev)
 *
 *  code is based of the Architecture proposed by: "https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/"
 */\n""")

############ importing all relevant modules ############
import mxnet as mx, os, numpy as np, matplotlib.pyplot as plt, cv2, logging, random, scipy, seaborn as sns, re, time
from PIL import Image
from shutil import copyfile
from unet import *
from imagefuncs import *
logging.getLogger().setLevel(logging.INFO) # log the training session
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

############ context ###################################
device_context = mx.gpu(0); """set device over which the training will be perfomed. use mx.cpu() for training over CPU cores, however,
the training over CPU will be considerably slower (3 hours on an 8 core machine) compared to 7 minutes on my GPU (640 CUDA cores).
Yes, the difference is astounding !""";

print("version of MXNET: ", mx.__version__ ," -> with context set to: ", device_context,"\n")

############# global constants & HyperParamaters ###################
width,height = (160,160)    #images and masks will be resized according to the following tuple
filtercount = 64
directory = 'C:\\Users\\aliha\\Downloads\\fabrice-ali\\deeplearning\\';
kernel_size = (3,3) #kernel size for convolutions
pad_size = (1,1) #padding for convolutions
initializer = mx.initializer.Normal(np.sqrt(2 / 576))
num_round = 10; #number of epochs (for training rounds)
batch_size = 8; #batch-size to process
fractionalp = 10/13; #training-dataset/testing-dataset ratio
lr = 0.01;  #learning rate
optimizer = 'adam' # other possible options are: 'adam', 'rmsprop', 'nadam' etc..
optimizerdict = {'learning_rate': lr}
train,retrain,applynet = (False,False,True)
# for retraining the network
(start_epoch,step_epochs) = (10,0)
save_round = start_epoch + step_epochs

############## setting directory #################
if(os.getcwd() != directory):
    os.chdir(directory)

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

training_image_directory = directory + "data\\train\\train_images_8bit\\";
imagefilenames = sorted_aphanumeric(os.listdir(training_image_directory)) # list of all images in the directory

training_label_directory = directory + "data\\train\\train_masks\\"
maskfilenames = sorted_aphanumeric(os.listdir(training_label_directory)) # list of all masks in the directory

###############  generate training and test datasets ###################

# load and resize labels -> ensure binary images
imagefilenames = list(map(lambda x: training_image_directory + x, imagefilenames))
maskfilenames = list(map(lambda x: training_label_directory + x, maskfilenames))
# considering a subset of data and corresponding labels
imagefilenames = imagefilenames[0:390]
maskfilenames = maskfilenames[0:390]

##### Resize images
train_x = np.asarray(list(map(lambda x: np.array(imageResize(x,width,height)), imagefilenames))) # """ same as below, just more functional in nature"""
#train_x = np.asarray([np.array(imageResize(imagefilenames[i],width,height)) for i in range(len(imagefilenames))])

##### image resizing with CLAHE OpenCV
#train_x = np.asarray(list(map(lambda x: np.array(claheResize(x,width,height)),imagefilenames)))
#train_x = np.asarray([np.array(claheResize(imagefilenames[i],width,height)) for i in range(len(imagefilenames))])

##### Resize Masks/Labels
train_y = np.asarray(list(map(lambda x: np.array(imageResize(x,width,height)),maskfilenames)))
#train_y = np.asarray([np.array(imageResize(maskfilenames[i],width,height)) for i in range(len(maskfilenames))])
train_y[train_y >= 1] = 1; # ensure binarization
train_y[train_y < 1] = 0;


# splitting datasets to training and testing halves
N = len(maskfilenames)
n = int(np.floor(N*fractionalp)) # adjust fractionalp to change the training/testing datasets lengths
print("length of training dataset:", n, " samples")
print("length of validation dataset:", N-n, " samples\n")

assert len(train_x) == len(train_y)
train_x = train_x.reshape((len(train_x),1,width,height)) # array reshaping required for data and labels
train_y = train_y.reshape((len(train_y),1,width,height))
train_x_array,test_x_array = (train_x[:n], train_x[n:])
train_y_array,test_y_array = (train_y[:n], train_y[n:])

#---------------------------------------- Learning Factory -----------------------------------------------

os.chdir(directory + "saved_models\\") # dir to save the training model

print("### Architecture of U-net ###\n")
######################## make the network ########################
net = get_unet(filtercount, kernel_size, pad_size, batch_size, width, height) # generate the symbolic network (uninitialized)
mx.viz.plot_network(net, save_format = 'pdf').render() # visualize the neural network -> check directory for save

# internal metrics can be used, in contrast build custom metrics if need-be
fig = plt.matshow(np.random.random((height,width)))

def custom_rmse(label,pred):
    #print("label dim: ", len(label), "  prediction dim: ", len(pred))
    for tensor in pred:
        fig.set_data(tensor[0])
        plt.draw()
        plt.pause(0.005)
    return np.sqrt(np.mean((label-pred)**2))

def custom_logloss(label,pred):
    return np.mean((label*np.log(pred)) + ((1-label)*np.log(1-pred)))

metric_custom_rmse = mx.metric.CustomMetric(feval = custom_rmse)
#metric_internal = mx.metric.create(['acc','rmse'])
#metric_logloss_Custom = mx.metric.CustomMetric(feval = custom_logloss)
#rounded_mean_err = lambda labels, predictors : np.mean(np.abs(labels-np.round(predictors)))
#metric_rmse_Custom_rounded = mx.metric.CustomMetric(feval = rounded_mean_err)

train_iter = mx.io.NDArrayIter(train_x_array, train_y_array, batch_size, label_name = 'target', shuffle=True)
#val_iter = mx.io.NDArrayIter(data = test_x_array, label = test_y_array, batch_size = batch_size, shuffle=True)
# not validating


# training the network-model
if train:
    mod = mx.mod.Module(symbol=net, data_names=['data'], label_names=['target'], context=device_context)
    mod.bind(data_shapes= train_iter.provide_data, label_shapes= train_iter.provide_label)
    mod.fit(
        train_data = train_iter,
        #eval_data = val_iter,
        optimizer= optimizer,
        initializer = initializer,
        optimizer_params=optimizerdict,
        eval_metric = metric_custom_rmse,
        num_epoch = num_round
        )
    #print(model.score(val_iter,eval_metric = metric_custom_rmse))
    mod.save_checkpoint("blobseg_model", num_round)

    # saving the trained model
    destination = directory + "saved_models\\lg_saves\\iteration " + str(num_round) + "\\";
    if not os.path.exists(destination):
        os.makedirs(destination)
    filesToCopy = [fname for fname in os.listdir() if fname.startswith("blobseg_model")]
    [copyfile(os.getcwd() + "\\" + fname, destination + fname) for fname in filesToCopy]


# --------------------------------------- Supplementary procedures ---------------------------------------------
############### retrain the network-model if necessary #################

if retrain:
    iteration = str(start_epoch)
    os.chdir(directory + "saved_models\\lg_saves\\iteration " + iteration)
    model_prefix = "blobseg_model"
    symbolicNet, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, start_epoch)
    model = mx.module.Module(symbol = symbolicNet,data_names=['data'],label_names=['target'],context = device_context)
    model.fit(
    train_data = train_iter,
    #eval_data = val_iter,
    arg_params = arg_params,
    aux_params = aux_params,
    optimizer = optimizer,
    num_epoch = step_epochs,
    optimizer_params = optimizerdict,
    eval_metric = metric_custom_rmse
    )
    # save the retrained net
    os.chdir(directory + "saved_models\\retrain\\")
    model.save_checkpoint(model_prefix,save_round)

############### load the pretrained network and apply over an image ################

############### load the pretrained network and apply over an image ################
randnum = np.random.randint(len(test_x_array))
if applynet:
    testimgdata = test_x_array[randnum].reshape((1,1,width,height)) # input data to the net
    fe_mod = loadNet(directory + "saved_models\\lg_saves\\iteration ", num_round, "blobseg_model", device_context,width,height)
    fe_mod.forward(Batch([mx.nd.array(testimgdata)]))         # apply the net on the input image
    features = fe_mod.get_outputs()[0].asnumpy()        # output tensor
    features[features >= 0.1] = 255        # assign 255 (white)
    plt.imshow(features[0][0],cmap = "gray")
    plt.show()
    plt.imshow(test_x_array[randnum,0],cmap="gray")
    plt.show()
    mask = np.array(features[0][0],dtype='uint8')       # peal tensor to get the matrix
    maskimg = Image.fromarray(mask)        # create image and display
    maskimg.save("C:/Users/aliha/Desktop/segmentationOutput.tif") # save segmentation mask
    img = Image.fromarray(test_x_array[randnum,0])        # create image and display
    img.save("C:/Users/aliha/Desktop/image_test.tif") # save segmentation mask

############### load the pretrained network and apply over a series of images (error was calculated below to check performance) ################

temp=[];
if applynet:
    fe_mod = loadNet(directory + "saved_models\\lg_saves\\iteration ", num_round, "blobseg_model", device_context, width, height)
    for j,i in enumerate(test_x_array[:10]): # test_x_array are the unseen images
        groundT = test_y_array[j]*1
        groundT[groundT >= 1] = 1
        groundT[groundT < 1] = 0
        i = i.reshape((1,1,width,height))
        fe_mod.forward(Batch([mx.nd.array(i)]))            # apply the net on the input image
        features = fe_mod.get_outputs()[0].asnumpy()       # output tensor
        features[features >= 0.1] = 1
        features[features < 0.1] = 0
        temp.append(1.0 - scipy.spatial.distance.hamming(groundT.flatten(),features.flatten()))
        mask = np.array(features[0,0],dtype='uint8')       # peal tensor to get the matrix
        maskimg = Image.fromarray(mask)      # create mask image
        #maskimg.save("C:/Users/aliha/Desktop/output/mask/segmentationOutput" + str(j) + ".tif") # save segmentation mask
        img = np.array(test_x_array[j,0],dtype='uint8')       # peal tensor to get the matrix
        img = Image.fromarray(img)      # create image
        #img.save("C:/Users/aliha/Desktop/output/image/serial" + str(j) + ".tif") # save image
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(test_x_array[j,0],cmap="gray")
        axarr[1].imshow(features[0,0],cmap= "gray")
        plt.show()
        plt.pause(0.5)
        plt.close()

    np.mean(temp)
    sns.distplot(temp, kde = False, color="#4CB391",bins=15)
    plt.show()
