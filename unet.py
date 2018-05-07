#################### NETGRAPH ########################
import mxnet as mx

def inferred_shape(net, batch_size,width,height):
    """ extracts dimensions/shapes of the various layers """
    _,out_shapes,__ = net.infer_shape(data = (batch_size,1,width,height))
    return out_shapes

def printshape(str,net,batchsize,width,height):
    """ wrapper for printing layer shapes """
    print(str, inferred_shape(net,batchsize,width,height))

def encoder_module(net, kernel_size, pad_size, filter_count, downpool = True):
                       """ as specified by the name of the function, for convolutions """
                       if downpool:
                           net = mx.symbol.Pooling(net, pool_type = "max", kernel = (2, 2), stride = (2, 2))

                       net = mx.symbol.Convolution(data = net, kernel = kernel_size, stride = (1,1),
                             pad = pad_size, num_filter = filter_count)
                       net = mx.symbol.Activation(net, act_type = "relu")
                       net = mx.symbol.BatchNorm(net)
                       net = mx.symbol.Convolution(data = net, kernel = kernel_size, stride = (1,1),
                                                pad = pad_size, num_filter = filter_count)
                       net = mx.symbol.Activation(net, act_type = "relu")
                       net = mx.symbol.BatchNorm(net)
                       return net

def decoder_module(input1,input2,filtercount,kernel_size,pad_size,downpool=False):
    net = mx.symbol.Deconvolution(input1, kernel = (2, 2), pad = (0, 0), stride = (2, 2),
                         num_filter = filtercount)
    net = mx.symbol.Concat(net,input2)
    net = encoder_module(net, kernel_size, pad_size, filtercount,downpool)
    return net


def get_unet(filtercount, kernel_size, pad_size, drop, batchsize,width,height):
        """ generate symbolic neural network -> U-net"""
        data = mx.symbol.Variable('data')
        target = mx.symbol.Variable('target')

        ## ------------------ commencing contraction phase --------------------
        enc1 = encoder_module(data, kernel_size, pad_size, filtercount, downpool = False)
        net = enc1
        printshape("@enc module_1: ", net, batchsize,width,height)
        enc2 = encoder_module(net, kernel_size, pad_size, filtercount * 2)
        net = enc2
        printshape("@enc module_2: ", net, batchsize,width,height)
        enc3 = encoder_module(net, kernel_size, pad_size, filtercount * 4)
        net = enc3
        printshape("@enc module_3: ", net, batchsize,width,height)
        enc4 = encoder_module(net, kernel_size, pad_size, filtercount * 8)
        net = enc4
        printshape("@enc module_4: ", net, batchsize,width,height)
        enc5 = encoder_module(net, kernel_size, pad_size, filtercount * 16)
        net = enc5
        printshape("@enc module_5: ", net, batchsize,width,height)
        ## -------------------- commencing expansion phase ---------------------
        # deconvole and catenate layers
        net = decoder_module(net,enc4,filtercount*8,kernel_size,pad_size)
        printshape("@decoder_module_1: ", net, batchsize,width,height)
        net = decoder_module(net,enc3,filtercount*4,kernel_size,pad_size)
        printshape("@decoder_module_2: ", net, batchsize,width,height)
        net = decoder_module(net,enc2,filtercount*2,kernel_size,pad_size)
        printshape("@decoder_module_3: ", net, batchsize,width,height)
        net = decoder_module(net,enc1,filtercount,kernel_size,pad_size)
        printshape("@decoder_module_4: ", net, batchsize,width,height)
        # output after decoder
        net = mx.symbol.Convolution(data = net, kernel = (1,1), stride = (1,1),
              pad = (0,0), num_filter = 1)
        printshape("@conv: ", net, batchsize,width,height)
        net = mx.symbol.LogisticRegressionOutput(data = net, label = target)
        printshape("@output: ", net, batchsize,width,height)
        print("\n")
        return net
