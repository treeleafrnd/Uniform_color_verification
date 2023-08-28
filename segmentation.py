# Necessary imports
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw
from utils import color_detect, apply_mask, segmentation, histogram, dominant_color


# taking in the image path
img_path = "data/images/sari_sample_6.png"

# opening and displaying the image
image = Image.open(img_path)
image.show()

# generated mask and boundary polygon coordinates
mask, polygon = segmentation(img_path)

# displaying the mask image
mask.show("Image Mask")

# applying mask on the image and displaying masked image
masked = apply_mask(image, mask)
masked.save("data/masked_images/masked_sample5.png", format="png")
masked.show("Masked Images")

# Histogram of original image and masked image.
histogram(image)
red,green,blue = histogram(masked)

# Dominant color
dominant_color(red, green, blue)


# # Drawing Polygons
# draw = ImageDraw.Draw((image_s), mode="I")
# draw.polygon(polygon1, outline=(0, 255, 0), width=3)
# image_s.show()


