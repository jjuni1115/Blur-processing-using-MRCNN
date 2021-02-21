import os
import sys
import random
import math
import re
import time
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import cv2
import argparse
from skimage import img_as_ubyte
# Root directory of the project
#ROOT_DIR = os.path.abspath("../../")
ROOT_DIR='/content/drive/MyDrive/mrcnn/Mask_RCNN-master/Mask_RCNN-master/'
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn.model import log

#from samples.balloon import balloon

#%matplotlib inline

# Directory to save logs and trained model
parser=argparse.ArgumentParser()
parser.add_argument('--input',required=True,help='input your video or image path')    # input data path
parser.add_argument('--weight',required=False,default="C:\\Users\\jjh\\PycharmProjects\\mrcnn\\Mask_RCNN-master\\Mask_RCNN-master\\logs\\mask_rcnn_name.h5",help='input your weight(model) path')  #input weight path
parser.add_argument('--output',required=False,default='./output',help='input output path')  #input output path
args=parser.parse_args()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


config = Config()

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

# setting class

class_names = ['BG']
class_names.append('face') 
class_names.append('license_plate')

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)




# Load weights
print("Loading weights ", args.weight)
model.load_weights(args.weight, by_name=True)

#video open
cap=cv2.VideoCapture(args.input)
if cap.isOpened()==False:
    print('Can\'t open this video')
    exit()
width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps=cap.get(cv2.CAP_PROP_FPS)
print('fps{0}'.format(fps))
codec=cv2.VideoWriter_fourcc(*'DIVX')  #setting codec
filename='output.mp4'  
output=cv2.VideoWriter(args.output,codec,fps,(int(width),int(height)))

while True:
    print('{0}/{1}'.format(cap.get(cv2.CAP_PROP_POS_FRAMES),cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    ret,frame=cap.read()
    if frame is None:
        break
    #imgframe=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    results=model.detect([frame],verbose=0)
    ax=get_ax(1)
    r=results[0]
    output_frame=visualize.detect_video(frame, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'], ax=ax,
                                title="Predictions")
    output.write(output_frame)

    if cv2.waitKey(1)==27:
        break;
cap.release()
output.release()


