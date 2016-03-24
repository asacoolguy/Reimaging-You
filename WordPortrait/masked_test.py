#!/usr/bin/env python2
"""
Masked wordcloud
================
Using a mask you can generate wordclouds in arbitrary shapes.
"""

from os import path
from PIL import Image
import numpy as np
import MySQLdb
import gc
from operator import itemgetter

from wordportrait import *

d = path.dirname(__file__)
STOPWORDS = set([x.strip() for x in open(os.path.join(d, 'stopwords')).read().split('\n')])


# Read the whole text.
#text = open(path.join(d, 'obama_speech.txt')).read()

# get the input
db = MySQLdb.connect(host="localhost", port=3306, user="root", passwd="localroot", db="test")
cursor = db.cursor()
cursor.execute("SELECT * FROM ethan_status")

# filter table into array of tuples 
results = cursor.fetchall()
words = []
# filter out stopwords
stopwords_lower = set(map(str.lower, STOPWORDS))
for row in results:
	word = row[2]
	freq = row[3]
	#ope = row[5]
	#ext = row[6]
	#neu = row[7]
	#agr = row[8]
	#con = row[9]
	if word.lower() in stopwords_lower:
		continue

	tup = (word,freq)
	words.append(tup)

words.sort(key=itemgetter(1), reverse=True)

i = 0
for row in words:
	i = i + 1
	print row
	if i > 0:
		break


# read the mask image
obama_mask = np.array(Image.open(path.join(d, "obama_threshold.jpg")))

wc = WordCloud(background_color="white", max_words=3000, mask=obama_mask, prefer_horizontal=1,
	min_font_size = 1,)
# generate word cloud
wc.generate_from_frequencies(words)

# store to file
wc.to_file(path.join(d, "test1.png"))

#wc.recolor(color_func = gradient_color_func2)
#wc.to_file(path.join(d, "test2.png"))

image = wc.to_image()
image.show()

# close database link
db.close()
gc.collect()