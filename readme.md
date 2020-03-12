# Final part of the project "SAR for Natural Calamities using Drones and Computer Vision"

Videos extracted based on ROI from the live video feed give a much accurate description of it but needs to be processed for final determination of victims.
A layer of Human Action Recognition was applied on the extracted videos to further filter out false positives and inaccuracies in the preceding model the results from this stage are then sorted and fed into the next layer in the pipeline
The results from the HAR layer are now being pre-processed for applying Human Pose Estimation algorithms
Pre-processed data will be used to train state-of-the-art models and carry out determination of help and non-help scenarios