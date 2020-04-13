[image1]: ./images/obamas.jpg "Obamas"
[image2]: ./images/obamas_result.png "Obamas_result"
[image3]: ./images/obamas_detected.png "Obamas Detect image"
[image4]: ./images/the_beatles_detected.png "The beatles detect images"
[image5]: ./images/the_beatles_result.png "The beatles result images"
# Facial Keypoint Detection
The goal of this project is to use computer vision techniques and deep learning architectures to build a facial keypoint detection system, to identify facial keypoints to be used for facial tracking, facial pose recognition, facial filters, and emotion recognition.
| Original Image            | Result Image                  |
| ------------------------- | ----------------------------- |
| ![Original Image][image1] | ![Final Result Image][image2] |
## Steps
  1. Detect all the faces in an image.
  2. Pre-process detected face images so that they are accepted as input to the trained network.
  3. Use the trained model to detect facial keypoints on the image.
  
 ## Code
 main.py contains the source code to process an image

 model.py contains the source code defining the network architecture  

 datasetloader.py contains the source code to load and preprocess the data

 train.py contains the source code for training and saving the network

## Model
You can download a pre-trained [model](https://github.com/youssefshoeb/Facial-Keypoint-Detection/releases/download/v1.0/keypoints_model_1.pt) and place it in the `saved_models` folder, or train your own network. The pre-trained model is a Convolutional Neural Network (CNN) based off the [ NaimishNet](https://arxiv.org/pdf/1710.00977.pdf) architecture, with a minor adjustment of adding 2D batch normalizations after each convolutional operations, and 1D batch normalizations after each fully connected layer. The pre-trained model is trained with the hyper-parameters set in the train.py file.
 ### Dependencies
- [OpenCV](http://opencv.org/)
- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [Torch](http://PyTorchpytorch.org)
- [Torchvision](https://pytorch.org/docs/stable/torchvision/index.html)
- [Pandas](https://pandas.pydata.org/)


 ## Data
 All of the data you'll need to train a neural network should be placed in the subdirectory `data`. To get the data, run the following commands in your terminal:

```
wget -P data/ https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip

unzip -n data/train-test-data.zip -d data
```
This facial keypoints dataset consists of 5770 color images. These images are separated into:
- 3462 training images.
- 2308 test images.
  
The datasetloader.py file can be used to pre-process the images for training and testing the network. 
The following transforms are applied to the data in the datasetloader.py file:
- **Normalize**: convert the color image to grayscale values with a range of [0,1] and normalize the keypoints to be in a range of about [-1, 1] 
- **Rescale**: rescale an image to the desired size of the network. 
- **RandomCrop**: crop the image randomly to a certain size.
- **ToTensor**: convert numpy images to torch images.


## Facial Keypoint Detection Processing Pipeline
### Step 1: Detect Faces
OpenCV's pre-trained Haar Cascade face detector (available in the `detector_architectures` folder) is used to detect all the faces in an image. 

![Obamas Detected][image3]

 ### Step 2 & 3: Detect Facial Keypoint
 Pre-process each detected face and feed it into the CNN facial keypoint detector, to get the 68 keypoints, with coordinates (x, y), for that face.
 
![Obama Final Result Image][image2]
