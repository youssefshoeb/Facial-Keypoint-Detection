import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
from model import Net
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = cv2.imread('images/the_beatles.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plot the image
fig = plt.figure(figsize=(9, 9))
plt.imshow(image)
plt.axis('off')
plt.show()

# detect faces
# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier(
    'detector_architectures/haarcascade_frontalface_default.xml')

# run the detector
faces = face_cascade.detectMultiScale(image, 1.3, 5)

# make a copy of the original image to plot detections on
image_with_detections = image.copy()

# loop over the detected faces, mark the image where each face is found
for (x, y, w, h) in faces:
    # draw a rectangle around each detected face
    cv2.rectangle(image_with_detections, (x, y), (x+w, y+h), (255, 0, 0), 3)

fig = plt.figure(figsize=(9, 9))
plt.imshow(image_with_detections)
plt.axis('off')
plt.show()

net = Net()

# load saved model parameters
net.load_state_dict(torch.load(
    'saved_models/keypoints_model_1.pt', map_location=device))

net.eval()
image_copy = np.copy(image)
fig = plt.figure(figsize=(10, 5))

# loop over the detected faces from your haar cascade
for i, (x, y, w, h) in enumerate(faces):

    # Select the region of interest that is the face in the image
    padding = 40
    roi = image_copy[y-padding:y+h+padding, x-padding:x+w+padding]

    # Convert the face region from RGB to grayscale
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # Normalize the grayscale image
    roi = roi / 255.0

    # Rescale the detected face
    roi = cv2.resize(roi, (224, 224))

    # Reshape the numpy image shape (H x W x C) into (C x H x W)
    torch_roi = roi.reshape(1, roi.shape[0], roi.shape[1], 1)
    torch_roi = torch_roi.transpose((0, 3, 1, 2))

    # convert numpy image to tensor
    torch_roi = torch.from_numpy(torch_roi)

    # convert images to FloatTensors
    torch_roi = torch_roi.type(torch.FloatTensor)

    # Make facial keypoint predictions using your loaded, trained network
    output_pts = net(torch_roi)

    # un-transform the predicted key_pts data
    predicted_key_pts = output_pts.data
    predicted_key_pts = predicted_key_pts.numpy()

    # reshape to 68 x 2 pts
    predicted_key_pts = predicted_key_pts[0].reshape((68, 2))

    # undo normalization of keypoints
    output_pts = predicted_key_pts*50.0+100.0

    # Display each detected face and the corresponding keypoints
    fig.add_subplot(1, len(faces), i+1)
    plt.imshow(roi, cmap='gray')
    plt.scatter(output_pts[:, 0], output_pts[:, 1], s=20, marker='.', c='m')
    plt.axis('off')

plt.show()
