# MNIST Handwritten Digits Classification

## Overview

This project involves building a neural network to classify handwritten digits from the MNIST dataset. The process includes data loading and transformation, exploratory data analysis (EDA), model building, training, evaluation, and improving the model for better accuracy.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Building](#model-building)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Model Improvement](#model-improvement)
- [Results and Visualizations](#results-and-visualizations)
- [Saving the Model](#saving-the-model)
- [Files Included](#files-included)
- [Conclusion](#conclusion)

## Environment Setup

To replicate the environment and run this project, follow these steps:

1. Install necessary libraries: torch, torchvision, matplotlib, numpy.
   ```bash
   pip install torch torchvision matplotlib numpy
   ```


## Data Preparation

### Load and Transform Dataset

- Load the MNIST dataset using `torchvision.datasets.MNIST`.
- Apply necessary transformations, such as converting images to tensors.
- Create DataLoader objects for both the training and testing datasets to enable batch processing.

## Exploratory Data Analysis

- Explore the dimensions and characteristics of the data.
- Visualize some sample images from the dataset to get an understanding of the data.

## Model Building

- Construct a neural network using the layers available in `torch.nn` and activation functions from `torch.nn.functional`.
- The architecture includes multiple linear layers with activation functions applied between them.

## Model Training and Evaluation

- Define an optimizer (e.g., Adam) and a loss function (e.g., CrossEntropyLoss).
- Train the neural network over a set number of epochs.
- Record the loss at each epoch and evaluate the model's performance.

## Model Improvement

- Introduce improvements to the model, such as dropout layers to prevent overfitting and batch normalization layers to stabilize and accelerate training.
- Re-train the improved model and compare its performance with the original model.

## Results and Visualizations

- Plot the training and validation loss history to visualize the learning process.
- Compare the accuracy and loss of the original and improved models.

## Saving the Model

- Save the trained models using `torch.save` for future use.

## Files Included

- `MNIST_Handwritten_Digits.ipynb`: Jupyter notebook with complete code and explanations.
- `Network.pth`: Saved model file for the basic neural network.
- `ImprovedNetwork.pth`: Saved model file for the improved neural network.

## Conclusion

This project demonstrates the process of building, training, and improving a neural network for classifying MNIST handwritten digits. The model improvements, including dropout and batch normalization, significantly enhance performance and accuracy. For detailed code implementation and further insights, refer to the Jupyter notebook (`MNIST_Handwritten_Digits.ipynb`) and accompanying files.
