# -*- coding: utf-8 -*-
"""
test change
Credit to Jason Brownlee https://github.com/jbrownlee
https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
"""

import os
from PIL import Image
import time
import cv2
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN

os.environ['TF_CUDNN_DETERMINISTIC']='1'

Image.MAX_IMAGE_PIXELS = None
from tkinter import Tk
from tkinter.filedialog import askdirectory


def draw_image_with_boxes(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
		# draw the dots
		for key, value in result['keypoints'].items():
			# create and draw dot
			dot = Circle(value, radius=2, color='red')
			ax.add_patch(dot)
	# show the plot
	pyplot.show()

def save_faces(file_location, slidename, filename, result_list):
	# load the image
    data = pyplot.imread(file_location)
    [max_height, max_width, rgb] = data.shape

    ax = pyplot.gca()
	# plot each face as a subplot
    for i in range(len(result_list)):
		# get coordinates
        left, top, width, height = result_list[i]['box']
        # zoom out
        outer_right, outer_bottom = min(left + 2*width, max_width-1), min(top + 2*height, max_height-1)
        outer_top = max(0, top-2*height)
        outer_left = max(0, left-2*width)

        #find inner rectangle coordinates
        inner_left = left-outer_left
        inner_top = top-outer_top
        inner_right = min(inner_left+width, max_width-1)
        inner_bottom = min(inner_top+height, max_height-1)

        # create the zoomed out image
        face_image = data[outer_top:outer_bottom, outer_left:outer_right]

        #plot image and get the context for drawing boxes
        pyplot.imshow(face_image)
        ax = pyplot.gca()

        #create box
        rect = Rectangle((inner_left,inner_top), inner_right-inner_left, inner_bottom-inner_top, fill=False, color='red')

        #draw box
        ax.add_patch(rect)

        subfilename = '%s [%d, %d, %d, %d].png' % (filename, top, left, width, height)
        print(subfilename)

        pyplot.gca().set_axis_off()
        pyplot.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)
        pyplot.margins(0,0)
        pyplot.gca().xaxis.set_major_locator(pyplot.NullLocator())
        pyplot.gca().yaxis.set_major_locator(pyplot.NullLocator())

        pyplot.savefig(os.path.join('faces detected', slidename, subfilename))

        print(os.path.join('faces detected', slidename, subfilename))
        pyplot.clf()

if __name__ == "__main__":

    root = askdirectory(title='Select Folder')
    detector = MTCNN()
    if not os.path.exists('faces detected'):
        os.makedirs('faces detected')

    for path, subdirs, files in os.walk(root):
        if not subdirs:
            starttime = time.time()
            slidename = path.split('/')[-1]
            slidename = slidename.split('\\')[-1]
            print(slidename)

            count = 0
            while os.path.exists(os.path.join('faces detected', slidename)):
                count = count + 1
                slidename = slidename.split()[0]
                slidename = '%s (%d)' % (slidename, count)
            if count == 0:
                os.makedirs(os.path.join('faces detected', slidename))
            else:
                os.makedirs(os.path.join('faces detected', slidename))


            for name in files:
                if name.endswith(".jpg"):
                    file_location = os.path.join(path,name)
                    filename = os.path.splitext(name)[0]

                    pixels = pyplot.imread(file_location)
                    faces = detector.detect_faces(pixels)

                    # draw_faces(file_location, faces)
                    save_faces(file_location, slidename, filename, faces)
                    # draw_image_with_boxes(file_location, faces)

            duration = time.time()-starttime

            print('Slide %s took %d seconds' % (slidename, duration))
