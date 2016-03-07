import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from skimage.data import camera
from skimage.filters import threshold_otsu


matplotlib.rcParams['font.size'] = 9


image = np.array(Image.open('vicki_test.jpg').convert('L'));
thresh = threshold_otsu(image)
binary = image > thresh

#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.5))
fig = plt.figure(figsize=(8, 2.5))
ax1 = plt.subplot(1, 3, 1, adjustable='box-forced')
ax3 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1, adjustable='box-forced')

ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Original')
ax1.axis('off')

ax3.imshow(binary, cmap=plt.cm.gray)
ax3.set_title('Thresholded')
ax3.axis('off')

plt.show()