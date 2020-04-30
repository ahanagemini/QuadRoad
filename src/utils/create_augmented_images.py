import os
import sys
from scipy import misc
import numpy
import random
import cv2
'''
Code to shift the intensity of each channel of tif image by x(=[-0.1....0.1])*std_dev where x is randomly selected  
Args: base_dir=the directory where images are stored
      filename= a list of the file names of the images for which computation is done
      save_dir: directory to save the augmented images
      img_type: hs, rgb or hght
'''
def save_shifted_image_hs(fname, base_dir, save_dir, shift_lst):
    """
    Function to create augmented image by shifting each channel
    for hyperspectral data
    Args:
        fname: name of file with list of tile names
        base_dir: directory with unaugmented images
        save_dir: Directory where we save original and shifted images
        shift_lst: a list to keep track of shift for each image
    Returns:
        shift_lst list of all shifts for images
    """
    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    # if shifts not yet created create them
    if len(shift_lst) == 0:
        find_shift = True
    else:
        find_shift = False
    #summation = numpy.zeros(3, dtype=numpy.float64)
    for i, tile in enumerate(tiles):
        image = cv2.imread(base_dir + "/" + tile + ".tif", -1)
        cv2.imwrite(save_dir + "/" + tile + ".tif", image)
        #image = image / 255
        num_pixel = image.shape[0] * image.shape[1]
        image = image.astype(float)
        sum_image = numpy.sum(image, axis=(0,1))
        #summation += sum_image
        sum_image = sum_image / num_pixel
        std_dev = find_std_dev(image, sum_image)
        # all channels of the 8 channel image is shifted by same 
        if find_shift:
            rand_shift = random.randint(-100, 100)
            shift = rand_shift/1000
            shift_lst.append(shift)
        else:
            shift = shift_lst[i]
        image = image + shift * std_dev
        image = image.astype('uint16')
        cv2.imwrite(save_dir+"/"+tile+"_s.tif",image)
    return shift_lst

def save_shifted_image(fname, base_dir, save_dir):
    """
    Function to create augmented image by shifting each channel
    for rgb and lidar data
    Args:
        fname: name of file with list of tile names
        base_dir: directory with unaugmented images
        save_dir: Directory where we save original and shifted images
    """

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    #summation = numpy.zeros(3, dtype=numpy.float64)
    for tile in tiles:
        image = misc.imread(base_dir + tile + ".png")
        to_save = misc.toimage(image, cmin=0,
                cmax=255).save(save_dir+tile+".png")
        image = image.astype(float)
        sum_image = numpy.sum(image, axis=(0,1))
        num_pixel = image.shape[0] * image.shape[1]
        #summation += sum_image
        sum_image = sum_image / num_pixel
        std_dev = find_std_dev(image, sum_image)
        rand_shift = random.randint(-100, 100)
        shift = rand_shift/1000
        image = image + shift*std_dev

def find_std_dev(image, means):
    """
    Function to compute std dev of image
    Args:
        image: the image being augmented
        means: list of the menas of each channel
    """
    if image.ndim == 3 and image.shape[2] == 3:
        std_dev = numpy.zeros(3, dtype=numpy.float64)
    else:
        std_dev = numpy.zeros(1, dtype=numpy.float64)
    dev_image = numpy.subtract(image, means)
    var = numpy.square(dev_image)
    num_pixel = image.shape[0] * image.shape[1]
    sum_var = numpy.sum(var, axis=(0,1))
    avg_var = sum_var / num_pixel
    std_dev = numpy.sqrt(avg_var)
    return std_dev    

if __name__ == "__main__":
    #Input graph files
    
    fname = sys.argv[1]
    base_dir = sys.argv[2]
    save_dir = sys.argv[3]
    img_type = sys.argv[4]
    if img_type == 'rgb' or img_type == 'hght':
        save_shifted_image(fname, base_dir, save_dir)
    if img_type == 'hs':
        shift_lst = []
        shift_lst = save_shifted_image_hs(fname, f'{base_dir}/123',
                f'{save_dir}/123', shift_lst)
        print("123 done")
        print(shift_lst)
        shift_lst = save_shifted_image_hs(fname, f'{base_dir}/456',
                f'{save_dir}/456', shift_lst)
        print("456 done")
        shift_lst = save_shifted_image_hs(fname, f'{base_dir}/78',
                f'{save_dir}/78', shift_lst)

