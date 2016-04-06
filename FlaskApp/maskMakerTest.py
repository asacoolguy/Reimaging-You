from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu, threshold_adaptive
from scipy.misc import toimage

# read image to array
image = np.array(Image.open('thresholdTest/rembrandt.jpg').convert('L'))
thresh = threshold_otsu(image)
binary = image > thresh
binary_adaptive = threshold_adaptive(image, method='mean',block_size=75, offset=10)

binary1 = toimage(binary)
binary2 = toimage(binary_adaptive)

binary1.save("normal_threshold.jpg")
binary2.save("adaptive_threshold.jpg")

binary2.paste(binary1, (0,0),binary1)

#img = Image.fromarray(binary,'RGB')
binary2.show()
binary2.save("combined_threshold.jpg")
#img = Image.fromarray(binary)

