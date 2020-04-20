######## Video Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/16/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a video.
# It draws boxes and scores around the objects of interest in each frame
# of the video.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

#Import packages

import os
import math
import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image
import glob
from skimage.measure import compare_ssim as ssim
import shutil
import os.path
from keras.preprocessing import image as Img
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'Test111.MP4'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 2
count=0

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()  #here1
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:  #here2
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)   #here3


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Open video file
video = cv2.VideoCapture(PATH_TO_VIDEO)

#delete
for file in os.listdir('C:/tensorflow1/models/research/object_detection/collection1/'):
    shutil.rmtree('C:/tensorflow1/models/research/object_detection/collection1/'+file)
files =glob.glob('C:/tensorflow1/models/research/object_detection/croped_image1/*')
for f in files:
    os.remove(f)
files =glob.glob('C:/tensorflow1/models/research/object_detection/video1/*')
for f in files:
    os.remove(f)
files =glob.glob('C:/tensorflow1/models/research/object_detection/testFinal/*')
for f in files:
    os.remove(f)


def rescale_list(input_list, size):
    assert len(input_list) >= size
    skip = len(input_list) // size
    output = [input_list[i] for i in range(0, len(input_list), skip)]
    return output[:size]
def action_recog(image_name):
    classes = glob.glob("train/*")
    classes = [classes[i].split('\\')[1] for i in range(len(classes))]
    classes = sorted(classes)
    
    cam = cv2.VideoCapture('video1/'+image_name)
    currentframe = 0
    print("Loading Model.......")
    model=load_model('checkpoints/lstm-features.120-0.259.hdf5')
    frames=[]
    while(True):
        ret,frame = cam.read()
        if ret:
            # if video is still left continue creating images
            name = 'testFinal/frame'+image_name +"frame_no"+ str(currentframe) + '.jpg'
            cv2.imwrite(name, frame)
            frames.append(name)
            currentframe += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
    rescaled_list = rescale_list(frames,10)

    base_model = InceptionV3(
        weights='imagenet',
        include_top=True
    )
    # We'll extract features at the final pool layer.
    inception_model = Model(
        inputs=base_model.input,
        outputs=base_model.get_layer('avg_pool').output
    )
    sequence = []
    for image in rescaled_list:
            img = Img.load_img(image, target_size=(299, 299))
            x = Img.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = inception_model.predict(x)
            sequence.append(features[0])

    sequence = np.array([sequence])
    prediction = model.predict(sequence)
    maxm = prediction[0][0]
    maxid = 0
    for i in range(len(prediction[0])):
          if(maxm<prediction[0][i]):
                maxm = prediction[0][i]
                maxid = i
    #print(frames)
    print(image_name,' ------- ',classes[maxid])
    return classes[maxid]

    
help_count=0


#standard dimension for croping
std_height=213
std_width=224
global fmt
fmt='.jpg'

def crop_and_save(frame,coordinates,count):
    img=frame
    num_of_detected_image=len(coordinates)
   
   
    for j in range(num_of_detected_image):
   
        croped=img[coordinates[j][0]:coordinates[j][1],coordinates[j][2]:coordinates[j][3]]
   
        #resizing to standard size
        croped_numpy=np.array(croped)
        croped_numpy= cv2.resize(croped_numpy, dsize=(std_height, std_width), interpolation=cv2.INTER_LINEAR)
        croped= Image.fromarray(croped_numpy, 'RGB')
   
        #save at desired location
        croped.save("croped_image1/frame"+str(count)+"croped"+str(j)+fmt)

#calculating mse
def MSE(imageA , imageB):
    err=np.sum((imageA.astype("float")- imageB.astype("float"))**2)
    err/=float(imageA.shape[0]*imageB.shape[1])
    return math.sqrt(err)
def compare_image(imageA,imageB):
    imageA=cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
    imageB=cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)
   
    m=MSE(imageA , imageB)

    s=ssim(imageA ,imageB)
    return m

count1=0
frames_images=[]
video_count=1
#directory = os.fsdecode('C:/tensorflow1/models/research/object_detection/Images1')
#for filename in glob.glob(directory+"/"+"*.jpg"):
while(video.isOpened()):
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    #frame=cv2.imread(filename)
    #frames_images.append(filename)
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
   

   
    #get the coordinates of the detected image of a frame
    coordinates = vis_util.return_coordinates(
                        frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.80)
    for i in range(len(coordinates)):
        crop_and_save(frame,coordinates,count)
   
    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.80)
       

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
    cv2.imwrite("pred11/image"+str(count1)+".jpg", frame)
   
    count1+=1
    count=count+1
    if(count==60):
        #files =glob.glob('C:/tensorflow1/models/research/object_detection/Images1/*')
        #for f in files:
         #   if(f in frames_images):
          #      os.remove(f)
        #frames_images=[]
        count=0
       
        #stacking
        images=[]
       
        for img in glob.glob("croped_image1/*.jpg"):
            img_cv= cv2.imread(img)
            images.append(np.array(img_cv))
       
        img_count=0
        stack_count=0
       
        if(len(images)>0):
            
            directory = os.fsdecode('C:/tensorflow1/models/research/object_detection/collection1')
            cv2.imwrite(directory+"/"+"stack"+str(stack_count)+"/"+str(img_count)+".jpg",images[0])
            img_count=img_count+1
            stack_count=stack_count+1
        check=0
        for img in images:
            min_err=10000
            print(check)
            check+=1
            flag=0
            for stack in os.listdir(directory):
                for file in os.listdir(directory+"/"+stack):
                    filename = os.fsdecode(file)
                    if(filename.endswith('.jpg')):
                        img_in_filename=Image.open(directory+"/"+stack+"/"+filename)
                        mse=compare_image(img,np.float32(img_in_filename))
                        if(mse<=42):
                            flag=1
                            if(min_err>mse):
                                min_err=mse
                                save=stack
                               
                           
            if(flag==1):
                cv2.imwrite(directory+"/"+save+"/"+str(img_count)+".jpg",img)
                img_count=img_count+1
               
            if(flag==0):
                stack_count=stack_count+1
                os.mkdir(directory+"/"+"stack"+str(stack_count))
                cv2.imwrite(directory+"/"+"stack"+str(stack_count)+"/"+str(img_count)+".jpg",img)
                img_count=img_count+1

        #deleting the stack having less then 10 frames        
        directory = os.fsdecode('C:/tensorflow1/models/research/object_detection/collection1')
        for stack in os.listdir(directory):
            file_count=0
            for file in os.listdir(directory+"/"+stack):
                file_count+=1
            #print(file_count)
            if(file_count<=10):
                shutil.rmtree(directory+"/"+stack)
               
        #video
        directory = os.fsdecode('C:/tensorflow1/models/research/object_detection/collection1')
       # video_count=1
       
       
        for stack in os.listdir(directory):
           
       
            img_array = []
            for filename in glob.glob(directory+"/"+stack+"/*.jpg"):
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)
         
         
               
            out = cv2.VideoWriter('C:/tensorflow1/models/research/object_detection/video1/'+str(video_count)+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
            video_count+=1
            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()  
        #prediction of video
        for file in os.listdir('C:/tensorflow1/models/research/object_detection/video1/'):
            out=action_recog(file)
            if(out=="Help"):
                help_count+=1
                shutil.copyfile("video1/"+file,"help_videos/"+str(help_count)+".mp4")
                shutil.copyfile("video1/"+file,"current_helpvideo/video.mp4")
                
        
            
       
       
       
       
       
        #delete
        #code for deleting
        for file in os.listdir('C:/tensorflow1/models/research/object_detection/collection1/'):
            shutil.rmtree('C:/tensorflow1/models/research/object_detection/collection1/'+file)
        files =glob.glob('C:/tensorflow1/models/research/object_detection/croped_image1/*')
        for f in files:
            os.remove(f)
        files =glob.glob('C:/tensorflow1/models/research/object_detection/video1/*')
        for f in files:
            os.remove(f)
        files =glob.glob('C:/tensorflow1/models/research/object_detection/testFinal/*')
        for f in files:
            os.remove(f)
        files =glob.glob('C:/tensorflow1/models/research/object_detection/pred1/*')
        for f in files:
            os.remove(f)
               
       
       
       
       
       
           
   
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()

