# Test part of the project "SAR using Drones and Computer Vision"


Please refer to this [link] for all the background information and documentation about the project


## Comments on the current status of project

This is the testing and evaluation part of our project. Initially we set out to complete all the objectives of the project and deploy for testing in a simulated environnment but due to the current coronavirus pandemic we were not able to procure the necessary equipment and also did not get the required computing power fo training the deep learning models on our custom dataset.
To compensate for this we created sample tests and extended that concept to a framework-independent model and pipeline evaluation system.

The AI pipeline for both the deployment approaches as well as the testing system is same on a broader sense of view.
The major changes in the testing system is that instead of working on videos we use images for all the predictions. This was done because of compatibility issues between windows and darknet which made us switch to a ubuntu based development environment, the latest version of opencv are not compatible with ubuntu but are required by the darknet system.
Before making the switch we evaluated both the pipeline approaches and did not find any significant performance difference.

## System requirements

- Ubuntu 18.04 (Highly recommended, setting up darknet on windows can be extremely difficult
- 10-12 GB of disc space
- CUDA enabled graphics card (recommended support is for CUDA 10 but system has been tested to work with CUDA 9 just fine)
- At least 8GB of RAM 

## Project setup

Significant modifications have been made to the source code of all dependencies so it is required to compile the system with our files.

### Download and rename as following: 

- [Darknet-based-YOLOv4] rename to "Darknet_Module"
- [RNN-for-HAR] rename to "HAR_Module"
- [Human-Action-Classification] rename to "Human-AC_Module"
- [tf-pose-estimation] rename to "TFPose_Module"

Now replace all the contents of the respective folders from our codebase with the ones inside the above renamed folders
Weight files and pre-trained models have to be downloaded from the above links as we cannot upload them here on github.

[link]: https://drive.google.com/drive/folders/1Ew4jQ_kBSTqr5Jz5T4HBjAJNhmoa-C1b?usp=sharing
[Darknet-based-YOLOv4]: https://github.com/AlexeyAB/darknet
[RNN-for-HAR]: https://github.com/stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input
[Human-Action-Classification]: https://github.com/dronefreak/human-action-classification
[tf-pose-estimation]: https://github.com/ildoonet/tf-pose-estimation