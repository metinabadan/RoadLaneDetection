import matplotlib.pyplot as plt
import cv2
import numpy as np

# show image in easy way
def showImage(window_name, image_name):
    cv2.imshow(window_name, image_name)
    cv2.waitKey(0)
    cv2.destroyAllWindows

def requiredPartOfImage(image, vertices):
    mask = np.zeros_like(image)
    matchMaskColor = 255
    cv2.fillPoly(mask,vertices,matchMaskColor)
    maskedImage = cv2.bitwise_and(image,mask)
    return maskedImage

# load an image
image = cv2.imread("road_lane.png")
#showImage("original image",image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# shape[0]  height  -  shape[1] width  -  shape[2] color channel
height = image.shape[0]
width = image.shape[1]

# to get lane part of image
verticesOfRequiredPartOfImage = \
    [
        (0,height),
        (width/2,height/2),
        (width,height)
    ]

grayImage = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
#showImage("gray",grayImage)

cannyImage = cv2.Canny(grayImage, 100, 200)
#showImage("canny",cannyImage)

croppedImage = requiredPartOfImage(cannyImage,np.array([verticesOfRequiredPartOfImage],np.int32))
#showImage("cropped",croppedImage)

# to see coordinates of pixels
plt.imshow(croppedImage)
plt.show()
