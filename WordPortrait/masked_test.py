#!/usr/bin/env python2
"""
Masked wordcloud
================
Using a mask you can generate wordclouds in arbitrary shapes.
"""

from os import path
from PIL import Image
import numpy as np

from wordportrait import *

d = path.dirname(__file__)

# Read the whole text.
text = open(path.join(d, 'obama_speech.txt')).read()


# read the mask image
# taken from
# http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpg
obama_mask = np.array(Image.open(path.join(d, "obama_threshold.jpg")))

wc = WordCloud(background_color="white", max_words=3000, mask=obama_mask, prefer_horizontal=1,
	min_font_size = 1,)
# generate word cloud
wc.generate(text)

# store to file
wc.to_file(path.join(d, "obama_cloud.png"))

image = wc.to_image()
image.show()
