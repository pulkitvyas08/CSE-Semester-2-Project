# Final part of the project "SAR for Natural Calamities using Drones and Computer Vision"

Videos extracted based on ROI from the live video feed give a much accurate description of it but needs to be processed for final determination of victims.
A layer of Human Action Recognition was applied on the extracted videos to further filter out false positives and inaccuracies in the preceding model the results from this stage are then sorted and fed into the next layer in the pipeline

The results from the HAR layer are now being pre-processed for applying Human Pose Estimation algorithms
Pre-processed data will be used to train state-of-the-art Human Pose Estimation (and, based on the need for more accuracy, action recognition) models and carry out determination of help and non-help scenarios

This entire AI pipline will then be deployed and installed on a Nvidia Jetson Nano board (or Raspberry Pi) on-board drone(s). Additionally a script will be written which will run the AI package and based on the final determination send the GPS coordinates of the location.

To test the accuracy, efficiency and speed of the system a UI is being developed using java which will allow the user to view the live camera feed, real-time Object Detection results, results after the Human Pose Estimation layer and final determination. 

Eventually the UI would be extended to be used in real-world situation where the user would be able to visualize the processing of individual drones and even swarms of drones (if possible by the internet conditions) and recieve the coordinates of locations where resue operations are needed to be done.