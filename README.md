# CNN-Classify_X_Ray_Images
## Project Description
In this project, I use Convolutional Neural Network to classify Pneumonia, a lounge illness, based on X-Ray images. I took 2 approaches, corresponding to 2 .py file in repository:
1. Build a simple CNN from scratch.
2. Import a Pre-Trained model and fine-tuned it.

## CONTEXT
Pneumonia is one of the leading respiratory illnesses worldwide, and its timely and accurate diagnosis is essential for effective treatment. Manually reviewing chest X-rays is a critical step in this process, and AI can provide valuable support by helping to expedite the assessment. In your role as a consultant data scientist, you will test the ability of a deep learning model to distinguish pneumonia cases from normal images of lungs in chest X-rays.

By fine-tuning a pre-trained convolutional neural network, specifically the ResNet-18 model, your task is to classify X-ray images into two categories: normal lungs and those affected by pneumonia. You can leverage its already trained weights and get an accurate classifier trained faster and with fewer resources.

### The Data
In this dataset of chest X-rays, there are 150 training images and 50 testing images for each category, NORMAL and PNEUMONIA (300 and 100 in total). For your convenience, this data has already been loaded into a `train_loader` and a `test_loader` using the `DataLoader` class from the PyTorch library. 
