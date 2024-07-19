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
1. labels are converted to **embeddings** using embedding layer
2. then each element in the embedding is expanded to image dimension so that embedding dimension becomes - batch_size x embdding_dim x img_size x img_size
3. that gets concatenated with image channel dimension and becomes - batch_size x (embdding_dim + channel_dim) x img_size x img_size. For MNIST channel_dim is 1
4. Then 1x1 convolution is used to bring the channel dimension back to 3 as the ResNet encoder block accepts 3 channels.
5. For decoder input, the output of **label embedding** is concatenated directly to latent dimension.
6. At the time of training, **label embedding** is used in encoder input and decoder input.

##### Training parameters

1. batch size: 512
2. learning rate: 1e-4
3. lr scheduler: None
4. precision: 32
5. epochs: 50

##### Output

![image](https://github.com/user-attachments/assets/a12397db-4cfa-46f1-967f-74d3219efd6c)

##### Observation

- Convergence of losses seems very difficult
- The output image is always decided by the input label instad of input image
- While expectation was to see a mix of input image and input label as output image, the outcome does not align to that.
- Therefore, the plan was to use half precision of speedy training, LR finder to accelerate loss convergence and more epochs to see if mixed up imaged can be generated.

#### Experiment 2: [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S22/blob/main/VAE_for_MNSIT_image_and_label_v2.ipynb)
Important features for this experiments are - 
1. labels are converted to **one-hot encoding**
2. then each element in the encoding dim (i.e. number of labels) is expanded to image dimension so that ecoding dimension becomes - batch_size x num_labels x img_size x img_size
3. that gets concatenated with image channel dimension and becomes - batch_size x (num_labels + channel_dim) x img_size x img_size. For MNIST channel_dim is 1
4. Then 1x1 convolution is used to bring the channel dimension back to 3 as the ResNet encoder block accepts 3 channels.
5. For decoder input, the output of **label encoding** is concatenated directly to latent dimension.
6. At the time of training, **label encoding** is used in encoder input and decoder input.

##### Training parameters

1. batch size: 512
2. learning rate: 1e-4
3. lr scheduler: None
4. precision: 32
5. epochs: 50

##### Output

![image](https://github.com/user-attachments/assets/29a0e9a2-3bfc-4cb8-b71c-2a47b4e9f1ea)

##### Observation

- Convergence of losses seems very difficult
- The output image is always decided by the input label instad of input image
- While expectation was to see a mix of input image and input label as output image, the outcome does not align to that.
- As the quality of these images are not as good as experiment 1, no further updates are done on this.

#### Experiment 3: [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S22/blob/main/VAE_for_MNSIT_image_and_label_v3.ipynb)
Important features for this experiments are - 
1. labels are converted to **one-hot encoding**
2. then each element in the encoding dim (i.e. number of labels) is expanded to image dimension so that ecoding dimension becomes - batch_size x num_labels x img_size x img_size
3. that gets concatenated with image channel dimension and becomes - batch_size x (num_labels + channel_dim) x img_size x img_size. For MNIST channel_dim is 1
4. Then 1x1 convolution is used to bring the channel dimension back to 3 as the ResNet encoder block accepts 3 channels.
5. For decoder input, labels are converted to **embeddings** using embedding layer and the the output of **label embedding** is concatenated directly to latent dimension.
6. At the time of training, **label encoding** is used in encoder input but **label embedding** is used for decoder input.

##### Training parameters

1. batch size: 512
2. learning rate: 1e-4
3. lr scheduler: None
4. precision: 32
5. epochs: 50

##### Output

![image](https://github.com/user-attachments/assets/445ecf47-a76b-4ff2-989b-82a04a837ad8)


##### Observation

- Convergence of losses seems very difficult
- The output image is **mostly** decided by the input label instad of input image
- While expectation was to see a mix of input image and input label as output image, the outcome does not align to that.
- As the quality of these images are not as good as experiment 1, no further updates are done on this.

#### Experiment 4: [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S22/blob/main/VAE_for_MNSIT_image_and_label_v1_2.ipynb)

Important features for this experiments are - 
1. This only an extension to Experiment1
2. Trained for longer time and used One-Cycle LR and half precision for faster iteration
![image](https://github.com/user-attachments/assets/96a9aa74-f853-4e63-a776-705653c62f44)

##### Training parameters

1. batch size: 512
2. Max learning rate: 0.001 (Tried using LR range test, however could not train with suggested LR, it was causing gradient explosion)
3. lr scheduler: One-Cycle LR
4. precision: 16
5. epochs: 100

##### Output
![image](https://github.com/user-attachments/assets/b15f9c1a-3ff1-4a47-b1af-d98e7378d094)

##### Loss curves

![image](https://github.com/user-attachments/assets/38c2718d-a4c1-4967-adb5-2ab634147d0a)

![image](https://github.com/user-attachments/assets/c4059aa1-7cef-46b8-9312-8cf0871d37e7)

![image](https://github.com/user-attachments/assets/fd642643-2e67-4549-923d-8502039aca93)

##### Observation

- All loss plateued
- The output image is decided by the input label instad of input image
- While expectation was to see a mix of input image and input label as output image, the outcome does not align to that.

#### Experiment 5: [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S22/blob/main/VAE_for_MNSIT_image_and_label_v3_1.ipynb)
Important features for this experiments are - 
1. This only an extension to Experiment3
2. Trained for longer time and used One-Cycle LR and half precision for faster iteration
![image](https://github.com/user-attachments/assets/fc039ea1-afd6-4104-9657-0fcbf7b5e621)

##### Training parameters

1. batch size: 512
2. Max learning rate: 0.02 (Used LR Range test and used 10 timed less than suggested LR)
3. lr scheduler: One-Cycle LR
4. precision: 16
5. epochs: 100

##### Output
![image](https://github.com/user-attachments/assets/84c4702f-5551-4b11-a3f4-035fe5850beb)


##### Loss curves
![image](https://github.com/user-attachments/assets/d840b85c-2d98-4a3b-ada1-e54149ae231f)

![image](https://github.com/user-attachments/assets/8c65d8a7-7623-483d-9c81-fce34aadb10b)

![image](https://github.com/user-attachments/assets/e075da50-3621-4c00-8922-a92f7670fec9)

##### Observation

- Losses are very jittery and no image got reconstructed
- Need further analysis to understand the issue


### CIFAR10 experiments

#### Experiment 1: [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S22/blob/main/VAE_for_CIFAR10_image_and_label_v1.ipynb)
Important features for this experiments are - 
1. labels are converted to **embeddings** using embedding layer
2. then each element in the embedding is expanded to image dimension so that embedding dimension becomes - batch_size x embdding_dim x img_size x img_size
3. that gets concatenated with image channel dimension and becomes - batch_size x (embdding_dim + channel_dim) x img_size x img_size. For CIFAR10 channel_dim is 3
4. Then 1x1 convolution is used to bring the channel dimension back to 3 as the ResNet encoder block accepts 3 channels.
5. For decoder input, the output of **label embedding** is concatenated directly to latent dimension.
6. At the time of training, **label embedding** is used in encoder input and decoder input.

##### Training parameters

1. batch size: 512
2. learning rate: 1e-4
3. lr scheduler: None
4. precision: 32
5. epochs: 50

##### Output

![image](https://github.com/user-attachments/assets/8c3593c9-1b07-434a-ad3b-af48218ee4b5)


##### Observation

- Convergence of losses seems very difficult
- Reconstruction of output image is very poor
- While expectation was to see a mix of input image and input label as output image, the outcome does not align to that.
- Therefore, the plan was to use half precision of speedy training, LR finder to accelerate loss convergence and more epochs to see if mixed up imaged can be generated.

#### Experiment 2: [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S22/blob/main/VAE_for_CIFAR10_image_and_label_v2.ipynb)
Important features for this experiments are - 
1. labels are converted to **one-hot encoding**
2. then each element in the encoding dim (i.e. number of labels) is expanded to image dimension so that ecoding dimension becomes - batch_size x num_labels x img_size x img_size
3. that gets concatenated with image channel dimension and becomes - batch_size x (num_labels + channel_dim) x img_size x img_size. For CIFAR10 channel_dim is 3
4. Then 1x1 convolution is used to bring the channel dimension back to 3 as the ResNet encoder block accepts 3 channels.
5. For decoder input, the output of **label encoding** is concatenated directly to latent dimension.
6. At the time of training, **label encoding** is used in encoder input and decoder input.

##### Training parameters

1. batch size: 512
2. learning rate: 1e-4
3. lr scheduler: None
4. precision: 32
5. epochs: 50

##### Output

![image](https://github.com/user-attachments/assets/839d56d0-5da2-458d-a71c-2b81806a817f)


##### Observation

- Convergence of losses seems very difficult
- Reconstruction of output image is very poor
- While expectation was to see a mix of input image and input label as output image, the outcome does not align to that.

#### Experiment 3: [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S22/blob/main/VAE_for_CIFAR10_image_and_label_v3.ipynb)
Important features for this experiments are - 
1. labels are converted to **one-hot encoding**
2. then each element in the encoding dim (i.e. number of labels) is expanded to image dimension so that ecoding dimension becomes - batch_size x num_labels x img_size x img_size
3. that gets concatenated with image channel dimension and becomes - batch_size x (num_labels + channel_dim) x img_size x img_size. For MNIST channel_dim is 1
4. Then 1x1 convolution is used to bring the channel dimension back to 3 as the ResNet encoder block accepts 3 channels.
5. For decoder input, labels are converted to **embeddings** using embedding layer and the the output of **label embedding** is concatenated directly to latent dimension.
6. At the time of training, **label encoding** is used in encoder input but **label embedding** is used for decoder input.

##### Training parameters

1. batch size: 512
2. learning rate: 1e-4
3. lr scheduler: None
4. precision: 32
5. epochs: 50

##### Output

![image](https://github.com/user-attachments/assets/445ecf47-a76b-4ff2-989b-82a04a837ad8)


##### Observation

- Convergence of losses seems very difficult
- The output image is **mostly** decided by the input label instad of input image
- While expectation was to see a mix of input image and input label as output image, the outcome does not align to that.
- As the quality of these images are not as good as experiment 1, no further updates are done on this.


#### Experiment 4: [notebook](https://github.com/sayanbanerjee32/TASI_ERAv2_S22/blob/main/VAE_for_CIFAR10_image_and_label_v1_2.ipynb)

![image](https://github.com/user-attachments/assets/46e7d7b2-825a-44ab-97a0-84b50345f7e3)

![image](https://github.com/user-attachments/assets/85fb6cdd-b66b-4cb2-9388-8dbf43d3fcdf)

![image](https://github.com/user-attachments/assets/a4540d9a-c891-4161-9e65-936e27abfd3f)

![image](https://github.com/user-attachments/assets/708d1f81-74db-405f-9322-2fc72c1c831e)










