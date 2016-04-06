#!/usr/bin/env python2
from os import path
from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu, threshold_adaptive
from scipy.misc import toimage
import MySQLdb
import gc
from operator import itemgetter
from wordportrait import *

d = path.dirname(__file__)
STOPWORDS = set([x.strip() for x in open(os.path.join(d, 'stopwords')).read().split('\n')])

# takes in an img, returns an img with all the white in that img converted to alpha
def addTransparency(img):
	pixdata = img.load()

	for y in xrange(img.size[1]):
	    for x in xrange(img.size[0]):
	        if pixdata[x, y] == (255, 255, 255, 255):
	            pixdata[x, y] = (255, 255, 255, 0)	

	return img

# takes in an image and resizes it so either its width or height is maxSize
def resizeImg(img, maxSize):
	# height is bigger
	if img.size[1] > img.size[0]:
		maxIndex = 1
		otherIndex = 0
	# width is bigger
	else:
		maxIndex = 0
		otherIndex = 1

	resizePercent = (maxSize / float(img.size[maxIndex]))
	resizedSide = int((float(img.size[otherIndex]) * float(resizePercent)))

	if maxIndex == 1:
		resizedImg = img.resize((resizedSide, maxSize))#, Image.ANTIALIAS)
	else:
		resizedImg = img.resize((maxSize, resizedSide))#, Image.ANTIALIAS)

	return resizedImg

# read image to array
def makeMask(filename, imgSize):
	# open the image, resize it and turn it to greyscale
	image = Image.open('thresholdTest/' + filename + '.jpg').convert('L')
	image = resizeImg(image, imgSize)
	greyscale = np.array(image)

	# compute the global and adaptive thresholds
	thresh = threshold_otsu(greyscale)
	binary = greyscale > thresh
	binary_adaptive = threshold_adaptive(greyscale, method='mean',block_size=75, offset=10)

	# turn white space into transparency
	normal_threshold = toimage(binary).convert("RGBA")
	adaptive_threshold = toimage(binary_adaptive).convert("RGBA")
	normal_threshold = addTransparency(normal_threshold)
	adaptive_threshold = addTransparency(adaptive_threshold)

	# save the files and overlay then
	normal_threshold.save('thresholdTest/adaptiveThreshold/' + filename + "_normal_threshold.jpg")
	adaptive_threshold.save('thresholdTest/adaptiveThreshold/' + filename + "_adaptive_threshold.jpg")
	adaptive_threshold.paste(normal_threshold, (0,0), normal_threshold)

	#adaptive_threshold.show()
	adaptive_threshold.save('thresholdTest/adaptiveThreshold/' + filename + "_combined_threshold.jpg")




# Read the whole text.
#text = open(path.join(d, 'obama_speech.txt')).read()

# get the input
db = MySQLdb.connect(host="localhost", port=3306, user="root", passwd="localroot", db="test")
cursor = db.cursor()
cursor.execute("SELECT * FROM ethan_status")

# filter table into lists of tuples 
results = cursor.fetchall()
words = []

stopwords_lower = set(map(str.lower, STOPWORDS))
for row in results:
	word = row[2]
	freq = row[3]
	ope = row[5]
	ext = row[6]
	neu = row[7]
	agr = row[8]
	con = row[9]
	# filter out stopwords
	if word.lower() in stopwords_lower:
		continue

	tup = (word, freq, ope, con, ext, agr, neu)
	words.append(tup)

# sort words from largest freq to smallest
words.sort(key=itemgetter(1), reverse=True)

# i = 0
# for row in words:
# 	i = i + 1
# 	print row
# 	if i > 0:
# 		break


# read the mask image

filename = "jolie"
makeMask(filename, 1500)
mask = np.array(Image.open("thresholdTest/adaptiveThreshold/" + filename + "_combined_threshold.jpg"))

wc = WordCloud(background_color="white", max_words=5000, mask=mask, prefer_horizontal=1,
	min_font_size = 1, upper_font_filter=float(1/5), lower_font_filter=float(1/10),
	color_func = gradient_color_func)
# generate word cloud
wc.generate_from_frequencies(words)

# store to file
wc.to_file("thresholdTest/adaptiveThreshold/" + filename + "_cloud.jpg")

#wc.recolor(color_func = gradient_color_func2)
#wc.to_file(path.join(d, "test2.png"))

image = wc.to_image()
image.show()

# close database link
db.close()
gc.collect()