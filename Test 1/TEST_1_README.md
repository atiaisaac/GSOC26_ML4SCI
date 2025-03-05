# Introduction

In this test, we are provided with 30,000 train and 7500 validation images of gravitational lensing effects to be classified into 1 of 3 classes i.e subhalo substructure, vortex and no substructure.
This problem can be tackled using a supervised learning approach as the labels can be extracted from the image folder names indicated in the notebook implementation.

# Approach

This notebook uses keras 3.0 model definition with pytorch dataloaders. The reason being that pytorch dataloaders contains rich methods for image augmentation without necessarily relying on external libraries. Aside that, there are extended 3rd party support (like albumentations) that allows for image augmentation across most image tasks domains.

## Data

The input images are 1-channel images of size 150x150. However, most if not all open source transfer learning models were trained using images with 3 or more channels. Hence the input images needs to be preprocessed to have 3 channels or the model of interest be modified to work with a single channel. My implementation uses the former.

## Model architecture

Following the implementation in the paper [Deep Learning the Morphology of Dark Matter Substructure](https://arxiv.org/abs/1909.07346),(Stephon et al. 2019),a Resnet50 model pretrained on imagenet data is finetuned with a base learning rate of 1e-4 - which is in turn reduced by a factor of 10 every 3 epochs should there be no improvement in the validation loss as shown in the paper. Training is allowed to prgress for 50 epochs A Resnet model is considered because of its ability to counter the vanishing gradient problem exhibited by some deep neural networks via skip connections. The fully-connected part of the network is replaced with one for predicting 3 classes using the softmax activation function -  implying that the problem is a multi-class classification problem. The Adam optimizer is used because of its ability to converge faster, adaptive learning rate and momentum and computational efficiency across a plethora of models. 

## Loss

The loss function used is Cross entropy which ensures that the model is predicting outputs with a probability distribution close to that of the original
targets.

## Evaluation metric

As indicated in the test instructions, the metric of choice for evaluating the performance of the model is th Area under ROC curve. AUC-ROC is a better indicator of model performance and shows a trade-off between precision and recall values - something "accuracy" does not take into account. 

## Augmentation scheme

Image translation and rotation are the two augmentations used in the notebook. 

# Results

As shown in the [train report here](https://wandb.ai/atiaisaac007/GSOC25_DeepLense_Test/reports/GSOC-2025-Test-1--VmlldzoxMTY1MDc4Mg?accessToken=xwpx7ordc1ezl2zjaj2ye007ceaq2kngi2xxqrwuq1biutwb8j8i1irvotvpsep2) the validation loss is very poor at the beginning stages of training, however as the training progresses and the learning rate is reduced following the optimizer strategy discussed in model architecture, the validation metrics improves drastically eventually converging at around the 28th epoch.


