from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu, threshold_adaptive
from scipy.misc import toimage

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

def makeMask(filename):
	# open the image, resize it and turn it to greyscale
	image = Image.open('thresholdTest/' + filename + '.jpg').convert('L')
	image = resizeImg(image, 1500)
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


