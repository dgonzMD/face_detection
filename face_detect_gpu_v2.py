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
from mtcnn.mtcnn import MTCNN

os.environ['TF_CUDNN_DETERMINISTIC']='1'

Image.MAX_IMAGE_PIXELS = None
from tkinter import Tk
from tkinter.filedialog import askdirectory

def save_faces(file_location, slidename, filename, result_list):
	# load the image
    data = cv2.imread(file_location)
	# plot each face as a subplot
    for i in range(len(result_list)):
		# get coordinates
        left, top, width, height = result_list[i]['box']
        right, bottom = left + width-1, top + height-1 #may need to change this later using min function
        top = max(0, top)
        left = max(0, left)

        subfilename = '%s [%d, %d, %d, %d].jpg' % (filename, top, left, bottom, right)
        print(subfilename)
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

                    pixels = cv2.imread(file_location)
                    faces = detector.detect_faces(pixels)

                    # draw_faces(file_location, faces)
                    save_faces(file_location, slidename, filename, faces)
                    # draw_image_with_boxes(file_location, faces)

            duration = time.time()-starttime

            print('Slide %s took %d seconds' % (slidename, duration))
