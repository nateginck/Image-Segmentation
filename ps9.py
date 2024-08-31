# define kmeans function
import numpy as np
import scipy
import os
import PIL
from PIL import Image
import matplotlib.pyplot as plt

from kmeans_single import kmeans_single
from kmeans_multiple import kmeans_multiple
from Segment_kmeans import Segment_kmeans

# read in files
image1 = Image.open(r'input/im1.jpg').resize((100, 100))
image1 = np.array(image1)

image2 = Image.open(r'input/im2.jpg').resize((100, 100))
image2 = np.array(image2)

image3 = Image.open(r'input/im3.png').resize((100, 100))
image3 = np.array(image3)

# convert to double and scale down values
image1 = image1 / 255.0
image2 = image2 / 255.0
image3 = image3 / 255.0

# define variables for functions
K = [3, 5, 7]
Iters = [7, 13, 20]
R = [5, 15, 30]

# create list for images
images = [image1, image2, image3]
image_names = ['image1', 'image2', 'image3']

# create an image for each combination of K, Iters, and R
for image, image_name in zip(images, image_names):
    for k in K:
        for iters in Iters:
            for r in R:
                # create image
                im_out = Segment_kmeans(image, k, iters, r)

                # save image
                plt.axis('off')
                plt.title(f'{image_name}. K = {k}, Iters = {iters}, R = {r}')
                plt.imshow((im_out * 255).astype(np.uint8))
                filename = f'output/{image_name}_K{k}_Iters{iters}_R{r}.png'
                plt.savefig(filename)
                plt.close()

