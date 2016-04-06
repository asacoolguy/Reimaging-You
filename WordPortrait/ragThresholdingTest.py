
from skimage import data, io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt

import numpy as np
from os import path
from PIL import Image

#img = data.coffee()
img = np.array(Image.open('thresholdTest/swift.jpg'))#.convert('L'))

labels1 = segmentation.slic(img, compactness=30, n_segments=400)
out1 = color.label2rgb(labels1, img, kind='avg')

g = graph.rag_mean_color(img, labels1)
labels2 = graph.cut_threshold(labels1, g, 29)
out2 = color.label2rgb(labels2, img, kind='avg')

plt.figure()
io.imshow(out1)
plt.figure()
io.imshow(out2)
io.show()