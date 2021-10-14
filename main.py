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

def drawLines(image, lines):
    image = np.copy(image)
    blankImage = np.zeros((image.shape[0], image.shape[1],3),dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blankImage,(x1,y1),(x2,y2),(0,255,0),thickness=10)

    image = cv2.addWeighted(image, 0.8, blankImage, 1, 0.0)
    return image

"""
dont need anymore after the processing video

# load an image
image = cv2.imread("road_lane.png")
#showImage("original image",image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
"""

def process(image):


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

    cannyImage = cv2.Canny(grayImage, 100, 120)
    #showImage("canny",cannyImage)

    croppedImage = requiredPartOfImage(cannyImage,np.array([verticesOfRequiredPartOfImage],np.int32))
    #showImage("cropped",croppedImage)

    lines = cv2.HoughLinesP(croppedImage,2,np.pi/60,160,np.array([]),40,100)

    imageWithLines = drawLines(image,lines)

    return imageWithLines

cap = cv2.VideoCapture("Lane.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow("frame",frame)
    if (cv2.waitKey(1) & 0xFF == ord("q")):
        break

cap.release()
cv2.destroyAllWindows()


# to see coordinates of pixels
#plt.imshow(imageWithLines)
#plt.show()
