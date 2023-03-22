# Introduction

In this test, we are provided with 30,000 train and 7500 validation images of gravitational lensing effects to be classified into 1 of 3 classes i.e subhalo substructure, vortex and no substructure.
This problem can be tackled using a supervised learning approach as the labels can be extracted from the image folder names indicated in the notebook implementation.

# Approach

## Model architecture
Following the implementation in the paper [Deep Learning the Morphology of Dark Matter Substructure](https://arxiv.org/abs/1909.07346),(Stephon et al. 2019),a Resnet50 model pretrained on imagenet data is finetuned with a fixed learning rate of 1e-4 as opposed to the learning rate plateau scheme
used in the paper. A Resnet model is considered because of its ability to counter the vanishing gradient problem exhibited by some deep neural networks via skip connections. The fully-connected part of the network is replaced with one for predicting 3 classes using the softmax activation function -  implying that the problem is a multi-class classification problem.

## Loss
The loss function used is Cross entropy which ensures that the model is predicting outputs with a probability distribution close to that of the original
targets.

## Evaluation metric

As indicated in the test instructions, the metric of choice for evaluating the performance of the model is th Area under ROC curve. AUC-ROC is a better indicator of model performance and shows a trade-off between precision and recall values - something "accuracy" does not take into account. 

## Augmentation scheme

Image translation and rotation are the two augmentations used in the notebook. 
