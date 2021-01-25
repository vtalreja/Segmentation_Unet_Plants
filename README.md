# Segmentation_Unet_Plants
Repository for semantic segmentation of plants using UNet architecture. In this repo, a segmentation model for a plant dataset has been implemented. The segmentation model has been implemented using a famous neural network architecture UNet. In this implementation, the UNet architecture is not totally symmetric (number of encoding blocks equal to decoding blocks) and so, in that way it can also be considered as a Fully Convolutional network (FCN) architecture. 

# Dataset
The dataset used is very small with only about 160 images. The ground truth labels are also provided for each image. It could be observed from the dataset that there were 2 kinds of plants, one kind of plants, where the leaves were more symmetric and consistent in shape, there were about 112 images of these kind. The rest 48 images were of different kind of plant where the leaves were little unsymmetrical. The dataset was shuffled using both kind of plants and splitting
it into training, validation and testing split of 80,10, and 10 percent respectively.

# Data Augmentation
Considering such a small dataset of only 160 images, using data augmentation was inevitable. The following data augmentations were applied for the dataset: a) Horizontal Flip b) Vertical Flip c) Rotation from -10 to 10. Data augmentation was performed using multiple methods listed below.
## Augmentation using Image Processing Library
In this method, each image is augmented using the above augmentations and stored on the system. So, basically increasing the size of the dataset from 160 to 1120. This kind of augmentation is a static transformation where there is less randomness.
## Augmentation using the Pytorch framework
In this augmentation method, the images are augmented on the fly and not stored in the system as in the above method. So, basically, each image in this method is augmented on the fly, which provides some kind of randomness to the data augmentation.

# Loss Function
Weighted combination of binary cross entropy loss and the Jaccard (IOU) loss has been used as the loss function.
