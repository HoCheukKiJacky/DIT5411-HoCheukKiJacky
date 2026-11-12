# Traditional Chinese Handwritten Character Recognition (THEi DIT5411)

Author: Ho Cheuk Ki Jacky  
Course: DIT5411 Machine Learning Assignment  

## Introduction

In this project, I used Python and TensorFlow/Keras to build an AI model that can recognize handwritten Traditional Chinese characters.

- Dataset Source: Traditional Chinese Handwriting Dataset (created using Tegaki).
- Dataset Scale: There are 13,065 different Chinese characters in total, and each character has around 50 image samples (original size: 300x300 pixels).
- Train / Test Split: For each character, I used the first 40 images for training and the remaining 10 images for testing.

## Data Preprocessing & Augmented Image Creation

Since 40 images per character are not enough to train a good model for 13,065 classes, I used OpenCV (cv2) to create more training data. Each original image was used to make 4 new images, increasing the training set from 40 to 200 images per character (over 2.6 million images in total).

- Grayscale & Resizing: All images were changed to grayscale and resized to 48x48 during OpenCV augmentation, and then standardized to 32x32 pixels before model training so the model can train faster without running out of GPU memory. The pixel values were also rescaled by dividing by 255.
- OpenCV Transformations (cv2.getRotationMatrix2D & cv2.warpAffine):
  - Rotation: Randomly rotated between -12 and +12 degrees.
  - Scaling: Randomly zoomed in or out between 0.85x and 1.15x.
  - Shearing: Slanted horizontally between -0.15 and +0.15 to look like natural slanted handwriting.
  - Rotation + Translation: Rotated and shifted left, right, up, or down by up to 3 pixels.

## AI Model Architectures & Testing Accuracy

Referring to the example from kimanalytics, I built three different CNN models with different layers to compare their results on the testing set. All three models end with an output layer of 13,065 neurons using Softmax activation.

- Model A (Standard): A 3-layer CNN with batch normalization, dropout, and global average pooling. The testing accuracy improved to 35.08%.

- Model B (Baseline): A simple 2-layer CNN without batch normalization or dropout. The testing accuracy was 26.47%.

- Model C (Best Model): A deeper 5-layer double-convolutional CNN with batch normalization, dropout, and Flatten() to keep the position of character strokes. This model got the highest testing accuracy of 85.92%.

## Training Observation & Fine-Tuning (Model C)

### Issue Observed
When training Model C for 12 epochs, the validation accuracy reached 78.97% at Epoch 11, but suddenly dropped to 11.23% at Epoch 12 (similar drops also happened at Epochs 4 and 8). Meanwhile, the training accuracy kept going up smoothly to 88.77%.

### Why This Happened
This was not overfitting because the accuracy bounced back higher after earlier drops. Because there are 13,065 classes and we used mixed-precision (float16) to train faster, the learning rate was a bit too large. This caused the model weights to jump past the best point every few epochs and messed up the BatchNormalization values during testing.

### How I Fixed It
First, I lowered the Adam learning rate to 0.00001 (1e-5) and trained Model C for 2 more epochs so it could learn slowly and steadily. Second, I added ModelCheckpoint to automatically save the model from the epoch with the highest validation accuracy.

### Conclusion
After fine-tuning, Model C recovered and reached a final testing accuracy of 85.92%, which is much higher than Model A (35.08%) and Model B (26.47%).

## References

1. Traditional Chinese Handwriting Dataset (AI-FREE-Team):
   https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset
2. Introduction to OpenCV (Darshita Jain):
   https://github.com/darshitajain/Introduction-to-OpenCV
3. Handwritten Digit Recognition using Keras and TensorFlow (Kimanalytics):
   https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow
