from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu
from scipy.misc import toimage

# read image to array
image = np.array(Image.open('clinton_original.png').convert('L'))
thresh = threshold_otsu(image)
binary = image > thresh

img = toimage(binary)
img.show()

img.save("clinton_threshold.jpg")
#img = Image.fromarray(binary)