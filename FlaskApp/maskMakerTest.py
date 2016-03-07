from PIL import Image
from pylab import *
from skimage.filters import threshold_otsu
from scipy.misc import toimage

# read image to array
image = array(Image.open('clinton_original.png').convert('L'))
thresh = threshold_otsu(image)
binary = image > thresh

img = toimage(binary)
img.show()

img.save("clinton_threshold.jpg")
#img = Image.fromarray(binary)