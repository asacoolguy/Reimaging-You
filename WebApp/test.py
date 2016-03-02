import numpy as np
from os import path
from PIL import Image

d = path.dirname(__file__)

alice_mask = np.array(Image.open(path.join(d,"alice_mask.png")))
burrito_mask = np.array(Image.open(path.join(d,"burrito.png")))

print alice_mask.shape
print burrito_mask.shape

print type(alice_mask)
print type(burrito_mask)