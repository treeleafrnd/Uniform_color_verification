import cv2 as cv
import pandas as pd
from utils import assign_color
import time


# Color code dataset
color_csv = pd.read_csv("data/color_codes.csv")

# Declaring the image path
image_path = "data/masked_images/masked_sample2.png"

start = time.time()

# reading an image
image = cv.imread(image_path)

image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
resized = cv.resize(image_rgb, (480, 480), interpolation=cv.INTER_AREA)

# creating a pixel dictionary to store pixels data

pixel_dict = {
    "position": [],
    "value": [],
    "color": [],
}
loop_start = time.time()
for row in range(resized.shape[0]):
    for column in range(resized.shape[1]):
        print("Pixel")

        pixel_position = (row, column)
        pixel_dict["position"].append(pixel_position)

        pixel_value = tuple(resized[row, column])
        pixel_dict["value"].append(pixel_value)

        pixel_color = color_csv["group"].iloc[assign_color(pixel_value, color_csv)]
        pixel_dict["color"].append(pixel_color)

loop_end = time.time()
print("Loop execution time:", loop_end - loop_start)

pixel_df = pd.DataFrame.from_dict(pixel_dict)
pixel_df.head(10)
pixel_df.to_csv("data/pixel_data_masked2.csv")

end = time.time()

print("Execution time:", end - start)

