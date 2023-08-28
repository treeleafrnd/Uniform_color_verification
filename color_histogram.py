import PIL.Image
import numpy as np
import cv2 as cv
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from utils import get_color, dominant_color, apply_mask, histogram


# Extracting color histograms


def color_hist(image):

    #resized = cv.resize(mask, image.size, interpolation=cv.INTER_AREA)

    image = np.array(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    rgb_planes=cv.split(image)


    # blue color histogram
    hist=cv.calcHist(rgb_planes, channels=[2], mask=None, histSize=[256],ranges=[0,256], accumulate=False )
    plt.figure(1)
    plt.hist(hist, bins=[x for x in range(0,256)], color="b")

    #green color histogram
    hist=cv.calcHist(rgb_planes, channels=[1], mask=None, histSize=[256],ranges=[0,256], accumulate= False)
    plt.figure(2)
    plt.hist(hist, bins=[x for x in range(0,256)], color="g")

    #red color histogram
    hist=cv.calcHist(rgb_planes, channels=[0], mask=None, histSize=[256],ranges=[0,256], accumulate= False)
    plt.figure(3)
    plt.hist(hist, bins=[x for x in range(0,256)], color="r")

    plt.show()







#######################################################################################

# img = Image.open("images/car.png")
# img.show()
# # color_hist(img)
# pillow_hist(img)