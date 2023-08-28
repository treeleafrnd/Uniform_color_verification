import time
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import collections
import seaborn

start = time.time()

# Reading pixel data from csv file.
pixel = pd.read_csv("data/pixel_data_32.csv")

# lit of all colors in an image
all_colors = pixel["color"].unique()
print(all_colors)

# total amount of pixels
total_pixels = len(pixel)
image_path = "data/images/coat_sample_3.png"
image = cv2.imread(image_path)

# count of pixels each colors
count = []
color_info = {}
for color in all_colors:
    color_count = len(pixel[pixel["color"] == color])
    count.append(color_count)
    color_info[color] = round((color_count / total_pixels) * 100, 2)
    print(f"{color}:", round((color_count / total_pixels) * 100, 2), "%")

color_info = dict(sorted(color_info.items(), key=lambda v: v[1], reverse=True))

top_3_color_info = {}
for i, (k, v) in enumerate(color_info.items()):
    if i >= 3:
        break
    text = str(k) + ":" + str(v)
    cv2.putText(image, text=text, fontScale=1.0, fontFace=cv2.FONT_HERSHEY_SIMPLEX, org=(5, 50*2*(i+1)), color=(0, 200, 255), thickness=2)

    top_3_color_info[k] = v

end = time.time()

print("Execution time:", end - start)

cv2.imshow("Image", image)

# drawing the pie chart
plt.pie(count, labels=all_colors, autopct="%1.1f%%")
plt.show()




