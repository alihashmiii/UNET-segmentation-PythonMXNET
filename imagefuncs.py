from PIL import Image

def claheResize(imagefile,width,height):
    """ contrast limited adaptive histogram equalization from OPENCV and image resizing """

    img = cv2.imread(imagefile,0)
    clahe = cv2.createCLAHE().apply(img)
    return Image.fromarray(clahe).resize((width,height),Image.ANTIALIAS)

def imageResize(imgfilename, width,height,is8bit = True):
    """ takes in an image filename and outputs a resized version of the image with antialiasing """
    if not(is8bit):
        img = cv2.imread(imgfilename,cv2.IMREAD_UNCHANGED)
        img = img.astype('uint8')
        img = Image.fromarray(img)
        return img.resize((width,height),Image.ANTIALIAS)
    else:
        img = Image.open(imgfilename)
        return img.resize((width,height),Image.ANTIALIAS)
