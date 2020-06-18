"""
Video splitter: takes video file name/path as argument and saves the frames of the video
in video_frames directory
"""

import cv2
import sys

input_video = sys.argv[1]
vidcap = cv2.VideoCapture(input_video)
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("video_frames/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1 
