
# Cats and Dogs Classification Model

This project implements a Convolutional Neural Network (CNN) model to classify images of cats and dogs. The model is built using TensorFlow and Keras, and can be tested using a Gradio interface.

## Dataset

The dataset used for training and validation is the [Dogs vs. Cats dataset](https://www.kaggle.com/datasets/vinitdesai564/cats-dogs-classification-dataset/data) from Kaggle. You can download the dataset from [this link](https://www.kaggle.com/datasets/vinitdesai564/cats-dogs-classification-dataset/data).

## Model Architecture

The model consists of several convolutional layers followed by max-pooling layers, and finally a dense layer to output the classification result. The model architecture is as follows:

- Conv2D -> MaxPooling2D
- Conv2D -> MaxPooling2D
- Conv2D -> MaxPooling2D
- Conv2D -> MaxPooling2D
- Flatten
- Dense
- Dropout
- Dense (Output layer)

## Access the model(test model) here
 Signup for hugging faces and test the model:https://huggingface.co/ 
 /n
 Test the model here:https://huggingface.co/spaces/Vinit710/classification_models
