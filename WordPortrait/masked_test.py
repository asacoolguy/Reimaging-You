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
def makeMask(filename, person, imgSize):
	# open the image, resize it and turn it to greyscale
	image = Image.open('photoOp/' + person + '/' + filename + '.jpg').convert('L')
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
	normal_threshold.save('photoOp/' + person + '/thresholds/' + filename + "_normal_threshold.jpg")
	adaptive_threshold.save('photoOp/' + person + '/thresholds/' + filename + "_adaptive_threshold.jpg")
	adaptive_threshold.paste(normal_threshold, (0,0), normal_threshold)

	#adaptive_threshold.show()
	adaptive_threshold.save('photoOp/' + person + '/thresholds/' + filename + "_combined_threshold.jpg")

# image that makes images in bulk
def makePortraitBulk(photoName, topNum, botNum, person, imgSize, personaltyScores, posScore, negScore, words):
	for i in range(botNum, topNum + 1):
		filename = photoName + str(i)
		makePortrait(filename, person, imgSize, personaltyScores, posScore, negScore, words)

	print "everything is done!"
	

def makePortrait(filename, person, imgSize, personaltyScores, posScore, negScore, words):
	# read the mask image
	makeMask(filename, person, imgSize)

	image = Image.open("photoOp/" + person + "/thresholds/" + filename + "_combined_threshold.jpg")
	resizedImage = resizeImg(image, imgSize)
	mask = np.array(resizedImage)

	wc = WordCloud(background_color="white", max_words = 6000, mask = mask, prefer_horizontal=1,
		min_font_size = 1, upper_font_filter=float(1/5), lower_font_filter=float(1/10),
		color_func = gradient_color_func, personality_score=personaltyScores,
		pos_score = posScore, neg_score = negScore)
	# generate word cloud
	wc.generate_from_frequencies(words)

	# store to file
	wc.to_file("photoOp/" + person + "/" + filename + "_cloud.jpg")

	image = wc.to_image()
	print filename + " is done!"

# Read the whole text.
#text = open(path.join(d, 'obama_speech.txt')).read()

# get the input
# db = MySQLdb.connect(host="localhost", port=3306, user="root", passwd="localroot", db="test")
# cursor = db.cursor()
# cursor.execute("SELECT * FROM ethan_status")

photoName = "grace"
person = "grace_ware"
topNum = 10
botNum = 10
imgSize = 1500


db = MySQLdb.connect(host="127.0.0.1", port=3307, user="ethan", passwd="YnN&Y,[h6X,[NKp&", db="ethan")
cursor = db.cursor()
cursor.execute("SELECT * FROM 1gram_" + person)

# filter table into lists of tuples 
results = cursor.fetchall()
words = []

personaltyScores = [0,0,0,0,0]
posScore = [0,0,0,0,0]
negScore = [0,0,0,0,0]

stopwords_lower = set(map(str.lower, STOPWORDS))
for row in results:
	word = row[2]
	freq = row[3]
	ope = row[5]
	ext = row[6]
	neu = row[7]
	agr = row[8]
	con = row[9]

	if (word == ":d"):
		word = ":D"
	if (word == "d:"):
		word = "D:"
	if (word =="xd"):
		word = "XD"

	# calculate personality scores
	# scores still count even if the word is filtered out
	group_norm = row[4]
	personaltyScores[0] += group_norm * ope
	personaltyScores[1] += group_norm * con
	personaltyScores[2] += group_norm * ext
	personaltyScores[3] += group_norm * agr
	personaltyScores[4] += group_norm * neu

	if ope > 0:
		posScore[0] += group_norm * ope
	else:
		negScore[0] += group_norm * ope
	if con > 0:
		posScore[1] += group_norm * con
	else:
		negScore[1] += group_norm * con
	if ext > 0:
		posScore[2] += group_norm * ext
	else:
		negScore[2] += group_norm * ext
	if agr > 0:
		posScore[3] += group_norm * agr
	else:
		negScore[3] += group_norm * agr
	if neu > 0:
		posScore[4] += group_norm * neu
	else:
		negScore[4] += group_norm * neu


	# filter out stopwords
	if word.lower() in stopwords_lower:
		continue

	tup = (word, freq, ope, con, ext, agr, neu)
	words.append(tup)

# sort words from largest freq to smallest
words.sort(key=itemgetter(1), reverse=True)

# normalize personality scores
maxScore = max(abs(i) for i in personaltyScores)
personaltyScores[:] = [i / maxScore for i in personaltyScores]
# print maxScore
maxScore1 = max(abs(i) for i in posScore)
maxScore2 = max(abs(i) for i in negScore)
maxScore3 = max(maxScore1, maxScore2)
posScore[:] = [i / maxScore3 for i in posScore]
negScore[:] = [i / maxScore3 for i in negScore]
print personaltyScores
print posScore
print negScore

# personaltyScores = [0.44, -0.046, -0.75, -0.47, 1.0]
# posScore = [.45, 0.46, 0.63, 0.60, 0.35]
# negScore = [-.35, -0.89, -0.32, -0.32, -0.83]



makePortraitBulk(photoName, topNum, botNum, person, imgSize, personaltyScores, posScore, negScore, words)
#makePortrait(photoName, person, imgSize, personaltyScores, posScore, negScore, words)


# close database link
db.close()
gc.collect()