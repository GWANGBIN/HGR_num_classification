import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf
import keras
from keras import applications
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
from keras import backend as K
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

img_col = 224
img_row = 224
img_channel = 3

model = load_model('gesture_model.h5')

recog_result = 3

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_view = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha = 0.03), cv2.COLORMAP_JET)
        depth_colormap = cv2.convertScaleAbs(depth_image, alpha=0.45)
        depth_colormap = depth_colormap[0:480, 80:560]
        depth_colormap = depth_colormap[0:420:, 30:450]
#        depth_colormap = cv2.flip(depth_colormap, 0)
        depth_colormap = cv2.flip(depth_colormap, 1)
        depth_colormap = cv2.medianBlur(depth_colormap, 5)
        _ , new_map = cv2.threshold(depth_colormap, 220, 255, cv2.THRESH_TOZERO_INV)
        new_map = cv2.bitwise_not(new_map)
        _ , new_map = cv2.threshold(new_map, 185, 255, cv2.THRESH_TOZERO_INV)
#        cv2.imshow('new map', new_map)
        new_map = cv2.resize(new_map, (img_col, img_row), interpolation = cv2.INTER_CUBIC)
        preprocess_view = np.asanyarray(cv2.resize(new_map, (1200, 900), interpolation = cv2.INTER_CUBIC))


        imnew = image.img_to_array(new_map)
        batch_input = np.zeros((1, ) + (img_col, img_row, img_channel), dtype = K.floatx())
        batch_input[0] = imnew
        yhat = model.predict(batch_input)
        if yhat[0][0]>0.9:
            recog_result = "five"
        elif yhat[0][1]>0.9:
            recog_result = "four"
        elif yhat[0][3]>0.9:
            recog_result = "one"
        elif yhat[0][4]>0.9:
            recog_result = "three"
        elif yhat[0][5]>0.9:
            recog_result = "two"
        elif yhat[0][6]>0.9:
            recog_result = "zero"
        else:
            recog_result = "Not Detected"


        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth view', depth_view)
        cv2.putText(preprocess_view, recog_result, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0))

        cv2.imshow('Preprocess view', preprocess_view)
        cv2.waitKey(1)



finally:

    # Stop streaming
	pipeline.stop()
