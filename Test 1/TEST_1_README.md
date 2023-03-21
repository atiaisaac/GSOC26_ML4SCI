# Introduction

In this test, we are provided with 30,000 train and 7500 validation images of gravitational lensing effects to be classified into 1 of 3 classes i.e subhalo substructure, vortex and no substructure.
This problem can be tackled using a supervised learning approach as the labels can be extracted from the image folder names as indicated in the notebook implementation.

# Approach

Following the implementation in the paper [Deep Learning the Morphology of Dark Matter Substructure](https://arxiv.org/abs/1909.07346),(Stephon et al. 2019),
a Resnet50 model pretrained using imagenet data is finetuned with a fixed learning rate of 1e-4 as opposed to the learning rate plateau scheme
used in the paper. A Resnet model is considered as a result of its ability to counter the vanishing gradient problem exhibited by some deep neural networks.
It does this via skip connections. Image translation and rotation augmentation schemes are used to augment the training set of 30,000 images while no 
augmentations are applied to the validation images. 

The loss function used is Cross entropy which ensures that the model is predicting outputs with a probability distribution close to that of the origianl
targets, and the metric to quantify the models prediction is Aread under
