# Segmentation_Unet_Plants
Repository for semantic segmentation of plants using UNet architecture

In this repo, a segmentation model for the given
dataset had to be implemented. The segmentation model has
been implemented using a famous neural network architecture
UNet. In this implementation, the UNet architecture is not to-
tally symmetric (number of encoding blocks equal to decoding
blocks) and so, in that way it can also be considered as a Fully
Convolutional network (FCN) architecture. Weighted combi-
nation of binary cross entropy loss and the Jaccard (IOU)
loss has been used as the loss function.
