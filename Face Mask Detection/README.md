# Face_Mask_Detection: Project Overview

* Trained and deployed a Sequential model that predicts whether faces in input images wear mask or not.
* Image Dataset for training is obtained through kaggle challenge.
* Prediction using model is done in live video feed from local webcam.
* Built a REST API using Flask for the model.
* Deployed the API in simple Flask website for video processing.

## Code and Resources Used

**Python Version:** 3.7
**Packages:** pandas,numpy,tensorflow,matplotlib,opencv,flask,json,pickle,keras
**For Requirements:** ```pip install -r requirements.txt```

## Data Preprocessing

The Entire Dataset from kaggle is of size 2.5GB with 6900 images and each face in all those images manually annotated under 12 classes

I took 3 of those classes
* "Face with Mask"
* "Face with no mask"
* "Face with other covering"

Converted them to 2 classes "Mask" and "No Mask". Croppped each faces from the image and splitted them into Train , Test Data

Other Preprocessing steps include:

* Used Image Data Generator and Augmented them.
* Resized them into 50x50
* Binarized each resized image (img/255)

## Model Building

Built a 4 layer convolutional Model



![Model architecture](https://github.com/idifro/Data-Science/blob/master/Face%20Mask%20Detection/Data%20Modelling/model_plot.png "Model Architecture")

## Model Performance

With 16 Batches and 30 epoches followed by a learning rate of 0.001

I achieved a
* __training accuracy:__ 85
* __testing accuracy:__ 87

##Productionization

In this step, I built a flask API endpoint that was hosted on a local webserver and deployed the same REST API in a simple Flask website that opens up browser webcam.

Live video's each frame is sent to the API in JSON encoded form whcih returns the face co-ordinates followed by Classification result. That is plotted in the video feed directly.

![Output image](https://github.com/idifro/Data-Science/blob/master/Face%20Mask%20Detection/Output/Screenshot%20(117).png "Output 1")

![Output Image 2](https://github.com/idifro/Data-Science/blob/master/Face%20Mask%20Detection/Output/Screenshot%20(116).png "Output 2")
