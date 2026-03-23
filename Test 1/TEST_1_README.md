# Introduction

In this test, we are provided with 30,000 train and 7500 validation images of gravitational lensing effects to be classified into 1 of 3 classes i.e subhalo substructure, vortex and no substructure.
This problem can be tackled using a supervised learning approach as the labels can be extracted from the image folder names indicated in the notebook implementation.

# Approach

This notebook uses keras 3.0 model definition with pytorch dataloaders. The reason being that pytorch dataloaders contains rich methods for image augmentation without necessarily relying on external libraries. Aside that, there are extended 3rd party support (like albumentations) that allows for image augmentation across most image tasks domains.

## Data

The input images are 1-channel images of size 150x150. However, most if not all open source transfer learning models were trained using images with 3 channels. Hence the input images needs to be preprocessed to have 3 channels or the model of interest be modified to work with a single channel. My implementation uses the former by repeating the channel dimension 3 times.

## Model architecture

An efficientnet model pretrained on imagenet data is finetuned with a base learning rate of 1e-4 - which is in turn reduced by a factor of 10 every 3 epochs should there be no improvement in the validation loss as shown in the paper. Training is allowed to prgress for 50 epochs. An efficientnet is considered becuase of its superior scaling ability i.e its ability to balance network depth, width and resolution in response to input size. Efficientnet is also designed to use much fewer parameters and this makes them inherently efficient and faster. The fully-connected part of the network is replaced with one for predicting 3 classes using the softmax activation function -  implying that the problem is a multi-class classification problem. The Adam optimizer is used because of its ability to converge faster, adaptive learning rate and momentum and computational efficiency across a plethora of models. To ensure that training does not get stuck in a local minima, we train with a cosine decay with warmup scheduler. This forces the learning rate to increase during the first few epochs to escape the local minima curve then reduce gradually to allow reaching convergence.

## Loss

The loss function used is Cross entropy which ensures that the model is predicting outputs with a probability distribution close to that of the original targets.

## Evaluation metric

As indicated in the test instructions, the metric of choice for evaluating the performance of the model is the Area under ROC curve. AUC-ROC is a better indicator of model performance and shows a trade-off between precision and recall values - something "accuracy" does not take into account. 

## Augmentation scheme

The augmentations used are mainly geometric in nature and includes a random horizontal flip, affine transformations such as rotation and translation are used as well as a perspective shift. Due to the single color channel nature of the dataset, no color augmentations were used.

# Results

The graphs show that as the training progressing, the validation loss and validation AUC values improve, eventually converging at around the 27th epoch. The high AUC value shows that our model is doing very well in differentiating between the 3 classes. Aside that, the ROC curve also indicates that our model exhibits a high true positive rate(model is predicting a large proportion of true positives) and very low false postive rate(model is hardly misclassifying negative samples as positives)

# Recommendations

While the evaluation metrics indicate that our efficientnet model succeeded in learning transferable features, experimenting with other models could squeeze out some extra bit of performance. Consider using either;
1. Resnet architecture
2. Vision transformer
3. Vision transformer with convolutions at the early layers instead of a patchify layer.
4. Group Equivariant Convolutional Neural Networks

