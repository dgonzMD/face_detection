# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 19:36:25 2020

@author: Danny
"""

import os
from PIL import Image
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN

Image.MAX_IMAGE_PIXELS = None
from tkinter import Tk
from tkinter.filedialog import askdirectory

def draw_image_with_boxes(file_location, result_list):
	# load the image
    data = pyplot.imread(file_location)
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

def draw_faces(file_location, result_list):
	# load the image
    data = pyplot.imread(file_location)
	# plot each face as a subplot
    for i in range(len(result_list)):
		# get coordinates
        left, top, width, height = result_list[i]['box']
        right, bottom = left + width, top + height
		# define subplot
        pyplot.subplot(1, len(result_list), i+1)
        pyplot.axis('off')
		# plot face
        pyplot.imshow(data[top:bottom, left:right])
	# show the plot
    pyplot.show()

def save_faces(file_location, slidename, filename, result_list):
	# load the image
    data = pyplot.imread(file_location)
	# plot each face as a subplot
    for i in range(len(result_list)):
		# get coordinates
        left, top, width, height = result_list[i]['box']
        right, bottom = left + width, top + height

        face_image = data[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        subfilename = '%s [%d, %d, %d, %d].jpg' % (filename, top, left, bottom, right)
        pil_image.save(os.path.join('faces detected', slidename, subfilename), 'JPEG')
        print(os.path.join('faces detected', slidename, subfilename), 'JPEG')

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
