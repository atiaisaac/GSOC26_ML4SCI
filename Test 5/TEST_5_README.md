# Approach

This is a binary classification problem aimed at distinguishing between gravitational lensing images and non-lensing images. The approach addresses the class imbalance inherent in the dataset through a combination of weighted sampling, focal loss, and balanced data augmentation.

## Data Preprocessing and Splitting

The dataset is split into train and validation sets using stratified splitting to ensure both splits maintain the original class distribution. A custom `MyScale()` function performs min-max normalization on each image independently, scaling pixel values to the range [0, 1], which helps preserve fine details in the images. Also the minority class is oversample using a __WeightedRamdomSampler__. This means that while we are not generating new minority classes, we are increasing the frequency at which the model sees a minority class in each mini-batch forward pass.

## Model architecture

A ResNet18 model pretrained on ImageNet is used as the feature encoder. The fully-connected classification head is removed, and the raw feature embeddings (512-dimensional vectors) are fed into a lightweight linear classifier with sigmoid activation for binary prediction. The classifier includes a dropout layer (p=0.2) to reduce overfitting. This two-stage architecture (encoder + classifier) allows for flexible feature learning while maintaining computational efficiency. 

## Loss

Sigmoid Focal Loss is used instead of standard binary cross-entropy. Focal loss is specifically designed to address class imbalance by down-weighting easy negative examples and focusing training on hard-to-classify samples. This is more effective than standard loss functions for imbalanced binary classification problems, helping the model learn discriminative features for the minority class.

## Evaluation metric

As indicated in the test instructions, the metric of choice for evaluating the performance of the model is th Area under ROC curve. AUC-ROC is a better indicator of model performance and shows a trade-off between precision and recall values - something "accuracy" does not take into account. 

## Augmentation scheme

A comprehensive augmentation pipeline is applied to training data to improve model robustness:
- **Horizontal and Vertical Flips**: Each applied with 50% probability to simulate different gravitational lensing orientations
- **Random Affine Transformations**: Random rotation (0-10 degrees) and scaling (1.0-1.2×) with 30% probability to handle geometric variations in lensing patterns

These augmentations are applied only to the training set. Validation and test sets use only the custom normalization without augmentation to ensure fair evaluation. 

## Handling Class Imbalance

Given the significant class imbalance in the dataset (gravitational lenses are the minority class), two complementary strategies are employed:

1. **Weighted Random Sampler**: During training, samples are weighted inversely proportional to their class frequency. This ensures that minority class samples appear more frequently in each batch, allowing the model to learn from more diverse lens examples.

2. **Sigmoid Focal Loss**: This loss function automatically down-weights easy-to-classify examples while emphasizing hard-negatives and positives. The combination of weighted sampling and focal loss creates a synergistic effect that significantly improves minority class performance.

## Training Strategy

- **Optimizer**: Adam with learning rate of 1e-3 for efficient convergence with adaptive learning rates
- **Learning Rate Scheduler**: Cosine annealing over the full training duration to gradually reduce learning rate
- **Batch Size**: 32
- **Epochs**: Variable (as specified during training)

# Observation

The combination of weighted sampling and focal loss somewhat addresses the class imbalance problem partially. The model achieves moderate performance on the minority class (gravitational lenses), as evidenced by its high recall. The precision however, points to a different interpretation that, the model often misclassifies negative samples as positive.

# Recommendation
 
For future improvements, consider:
1. Ensemble methods combining multiple ResNet variants
2. Minority sample generation using GAN
3. Pretraining using a supervised contrastive approach for binary data imbalance then finetuning on a downstream classification task.

