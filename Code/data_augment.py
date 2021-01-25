import cv2
import skimage as sk
from skimage import transform
from skimage import util
import numpy as np
from skimage import io
from skimage.transform import rotate, AffineTransform, warp
import matplotlib.pyplot as plt
import random
from skimage import img_as_ubyte
import os
from PIL import Image
from skimage.util import random_noise


def rotation(image, angle):
    return rotate(image, angle)


def horizontal_flip(image):
    return np.fliplr(image)


def vertical_flip(image):
    return np.flipud(image)


def augmentation_function(image_path,augment_data_path):
    if not (os.path.exists(augment_data_path)):
        os.makedirs(augment_data_path)
    angles = [-10, -5, 5, 10]
    transformations = {
        'rotate': rotate,
        'horizontal_flip': horizontal_flip,
        'vertical_flip': vertical_flip }
    for img in os.listdir(image_path):
        image = os.path.join(image_path, img)
        image_name = img.split('.')[0]
        original_image = io.imread(image)
        for i in range(len(list(transformations))):
            key = list(transformations)[i]
            if key == "rotate":
                for angle in angles:
                    new_image_path = "%s/%s_augmented_image_%s_%s.png" % (augment_data_path, image_name, key, angle)
                    transformed_image = transformations[key](original_image, angle)
                    io.imsave(new_image_path, transformed_image)
            else:
                new_image_path = "%s/%s_augmented_image_%s.png" % (augment_data_path, image_name, key)
                transformed_image = transformations[key](original_image)
                io.imsave(new_image_path,transformed_image)

def convert_palette(image_path,image_name,dest_path):
    if not (os.path.exists(dest_path)):
        os.makedirs(dest_path)
    label = Image.open(image_path).convert('P')
    p = label.getpalette()
    label = np.asarray(label)
    img = Image.fromarray(label, mode='P')
    img.putpalette(p)
    img.save(os.path.join(dest_path,image_name))



if __name__ == '__main__':


    #original Image and Label paths
    image_path="/home/vtalreja/Documents/Signify_interview/images"
    label_path="/home/vtalreja/Documents/Signify_interview/labels"

    #Augmented Image and Label paths
    augment_data_path="/home/vtalreja/Documents/Signify_interview/images_augmented_1"
    augment_label_path="/home/vtalreja/Documents/Signify_interview/labels_augmented_1"

    # The labels converted in the above augmentation are in RGBA format but we need them to be
    # in the original Palette format. So after augmenting the dataset, we convert them into Palette format
    # and store it in this given directory and these are the final labels used in the segmentation model
    final_aug_label_path="/home/vtalreja/Documents/Signify_interview/labels_augmented_1_Palette"



    augmentation_function(image_path,augment_data_path)
    augmentation_function(label_path,augment_label_path)
    for img in os.listdir(augment_label_path):
        img_path = os.path.join(augment_label_path, img)
        convert_palette(img_path,img,final_aug_label_path)