
from typing import List
import PIL
from PIL import Image
import sys

# Image I/O
import imageio

import cv2
import os

import matplotlib.pyplot as plt

# Arrangement processing
import numpy as np
# Plots
import matplotlib.pyplot as pp
import matplotlib.patches as patches


# Image processing
from skimage import color, img_as_float, exposure, filters, measure, data


from methods import filters as ft
from methods import convolution_filters as conv


# Print shape and data type of images

print(f'\nRgb shape {imagem.shape}, size {len(imagem)} and type {imagem.dtype}\n')
print(f'Gray shape {imagemCinza.shape}, size {len(imagemCinza)} and type {imagemCinza.dtype}', end='\n' * 2)

# Histogram equalization

imagem_hist = exposure.equalize_hist(imagemCinza)

# Image segmented and otsu threshold

imagem_threshold = filters.threshold_otsu(imagem_hist)

print(f'Imagem threshold {imagem_threshold}', end='\n\n')

# Select the objects using the mask
imagem_segmented = imagem_hist > imagem_threshold

# Get labels

imagem_label = measure.label(imagem_segmented)

# Plot objects

fig, ax = pp.subplots(figsize=(10, 6))

ax.imshow(imagem, cmap='gray')  # Imagem

ax.set_title('Imagem em objetos')

ax.axis('off')

for region in measure.regionprops(imagem_label):
    # Take large regions
    if region.area > 50:
        minr, minc, maxr, maxc = region.bbox
        rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='blue')
        ax.add_patch(rect)
pp.show()
