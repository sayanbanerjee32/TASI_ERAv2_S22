# TASI_ERAv2_S22

## Objective

1. Design a variation of a VAE that takes in two inputs:
 - an MNIST image,
 - and its label (one hot encoded vector sent through an embedding layer)
2. Training as one would train a VAE
3. Now randomly send an MNIST image, but with a wrong label. Do this 25 times, and share what the VAE makes (25 images stacked in 1 image)!
4. Now replicate the same for CIFAR10 and share 25 images (1 stacked image)!

## Refernce Notebook
All experiments are performed by updated the this [collab notebook](https://colab.research.google.com/drive/1_yGmk8ahWhDs23U4mpplBFa-39fsEJoT?usp=sharing) that demostrates VAE training on CIFAR10 images using Pytorch Lightning 

## Experiemnts

The problem statement is perceived as a problem of Conditional VAE. The initial experimental thouthts are borrwed from this [article](https://towardsdatascience.com/conditional-variational-autoencoders-with-learnable-conditional-embeddings-e22ee5359a2a)

### MNIST experiments

#### Experiment 1: [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S22/blob/main/VAE_for_MNSIT_image_and_label_v1.ipynb)
Important features for this experiments are - 
1. labels are converted to embeddings using embedding layer
2. then each element in the embedding is expanded to image dimension so that embedding dimension becomes - batch_size x embdding_dim x img_size x img_size
3. that gets concatenated with image channel dimension and becomes - batch_size x (embdding_dim + channel_dim) x img_size x img_size. For MNIST channel_dim is 1
4. Then 1x1 convolution is used to bring the channel dimension back to 3 as the ResNet encoder block accepts 3 channels.
5. For decoder input, the output of label embedding is concatenated directly to latent dimension.
6. At the time of training, label encoding is used in encoder input and decoder input. 
