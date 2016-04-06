import matplotlib.pyplot as plt

from skimage import data
from skimage.filters import threshold_otsu, threshold_adaptive

from os import path
from PIL import Image
import numpy as np

#image = data.page()
image = np.array(Image.open('thresholdTest/rembrandt.jpg').convert('L'))

global_thresh = threshold_otsu(image)
binary_global = image > global_thresh

binary_adaptive = threshold_adaptive(image, method='mean',block_size=75, offset=10)

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2 = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Image')

ax1.imshow(binary_global)
ax1.set_title('Global thresholding')

ax2.imshow(binary_adaptive)
ax2.set_title('Adaptive thresholding')

for ax in axes:
    ax.axis('off')

plt.show()