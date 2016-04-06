from PIL import Image
import numpy as np
from scipy.misc import toimage


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

filename = "swift"

image = Image.open('thresholdTest/' + filename + '.jpg').convert('L')

resizedImage = resizeImg(image, 500)
#resizedImage.save('thresholdTest/adaptiveThreshold/' + filename + "_resized.jpg")


#img = Image.fromarray(binary,'RGB')
resizedImage.show()
resizedImage.save('thresholdTest/adaptiveThreshold/' + filename + "_combined_threshold.jpg")
#img = Image.fromarray(binary)

