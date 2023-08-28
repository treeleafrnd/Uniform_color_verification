import cv2 as cv
import numpy as np
import  PIL
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

def get_pixel_value(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        print(f"Index[x,y]:[{x},{y}]")
        print(f"(B,G,R):{img_rgb[x, y]}")
        # cv.putText(img_rgb,f"(B,G,R):{img_rgb[x,y]}",(10,10),cv.FONT_HERSHEY_SIMPLEX, 1.0, color=(255,255,255), thickness=3)
        cv.imshow("Image", img_rgb)


def color_detect(image):
    global img_rgb
    img = np.array(image)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    print(img_rgb.shape)
    cv.imshow("Image", img_rgb)
    cv.setMouseCallback("Image", get_pixel_value)

    cv.waitKey(0)

def get_color(num):
    if 0 <= num < 256:
        return "red"
    elif 256 <= num < 512:
        return "green"
    else:
        return "blue"

def dominant_color(red, green, blue):

    red = red[200:256]
    blue = blue[200:256]
    green = green[200:256]

    total_red = np.array(red).sum()
    total_green = np.array(green).sum()
    total_blue = np.array(blue).sum()
    print("total red", total_red)
    print("total green", total_green)
    print("total blue", total_blue)

    dominant= max(total_red,total_blue,total_green)

    print("Dominant", dominant)

    if total_red == dominant:
        print("Red color dominant")
    elif total_green == dominant:
        print("Green color dominant")
    else:
        print("Blue color dominant")

def apply_mask(image, mask):
    '''
    :param image: input image
    :param mask: mask to be applied
    :return: Returns the masked image
    '''
    resized_mask = mask.resize(image.size, resample=PIL.Image.NEAREST)
    resized_mask.show("Resized Mask")

    new_mask = resized_mask.convert("L")
    masked_image = Image.new('RGBA', image.size)
    masked_image.paste(image, (0, 0), new_mask)

    return masked_image

def histogram(image):
    # Histogram of the image
    hist = image.histogram()

    # Red color
    red_hist = hist[0:256]

    # Green color
    green_hist = hist[256:512]

    # blue color
    blue_hist = hist[512:768]

    # all
    plt.figure(1)
    for i in range(0, 768):
        plt.bar(i, hist[i], width=0.4, color=get_color(i), alpha=0.7)

    # red hist
    plt.figure(2)
    for i in range(0, 256):
        plt.bar(i, red_hist[i], width=0.4, color="red", alpha=0.7)

    # green hist
    plt.figure(3)
    for i in range(0, 256):
        plt.bar(i, green_hist[i], width=0.4, color="green", alpha=0.7)

    # blue hist
    plt.figure(4)
    for i in range(0, 256):
        plt.bar(i, blue_hist[i], width=0.4, color="blue", alpha=0.7)

    plt.show()
    return red_hist, green_hist, blue_hist


def segmentation(path):
    # selecting the model
    model = YOLO("yolov8m-seg.pt")

    # using the model to make predictions
    prediction = model.predict(path)

    result1 = prediction[0]

    # Identifying Masks
    masks = result1.masks

    # Only one mask is present
    mask1 = masks[0]
    mask = mask1.data[0].numpy()
    polygon = mask1.xy[0]
    mask_img = Image.fromarray(mask, "I")

    return mask_img, polygon

def get_rgb(hex):
    rgb_list = []
    for i in range(1,6,2):
        rgb_list.append(int(hex[i:i+2],16))

    return rgb_list



def assign_color(pixel, colors):
    pixel_value = pixel
    colors = colors["Hex"]
    distance = []

    for color in colors: # looping over the each colors of dataset.
        w_dist = 0
        rgb = get_rgb(color)

        for j in range(len(pixel_value)): # looping over each rgb channel.
             dist = abs(int(rgb[j]) - int(pixel_value[j]))
             w_dist += dist

        distance.append(w_dist)

    min_dist = min(distance)
    min_dist_index = distance.index(min_dist)

    return min_dist_index








