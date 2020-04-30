import os
import sys
from scipy import misc
import numpy
import tifffile
'''
Code to find the mean and std. dev. of  number of rgb images channel-wise
Args: base_dir=the directory where images are stored
      filename= a list of the file names of the images for which computation is done
      img_type: hs or rgb
'''
def find_mean_rgb(fname, base_dir):
    """
    Function to compute mean of RGB image
    Args:
        base_dir: the directory where images are stored
        filename: file with list of tile names
    Return:
        means of each channel
    """
    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    summation = numpy.zeros(3, dtype=numpy.float64)
    num_pixels = 0
    for tile in tiles:
        image = misc.imread(base_dir + tile + ".png")
        image = image.astype(float)
        image = image / 255
        num_pixel = image.shape[0] * image.shape[1]
        sum_image = numpy.sum(image, axis=(0,1))
        summation += sum_image
    
    summation = summation/(num_pixel * len(tiles))
    print("Mean")
    print(summation)
    return summation

def find_std_dev_rgb(fname, base_dir, means):
    """
    Function to compute std dev of RGB image
    Args:
        base_dir: the directory where images are stored
        filename: file with list of tile names
        means: list of the menas of each channel
    """
    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    std_dev = numpy.zeros(3, dtype=numpy.float64)
    num_pixel = 0
    for tile in tiles:
        image = misc.imread(base_dir + tile + ".png")
        image = image.astype(float)
        image = image / 255
        num_pixel = image.shape[0] * image.shape[1]
        dev_image = numpy.subtract(image, means)
        var = numpy.square(dev_image)
        sum_var = numpy.sum(var, axis=(0,1))
        std_dev = std_dev + sum_var
        
    std_dev = std_dev/(num_pixel * len(tiles))
    std_dev = numpy.sqrt(std_dev)
    print("Std. Dev.")
    print(std_dev)    

def find_mean_hs(fname, base_dir):
    """
    Function to compute mean of hyper-spectral image
    Args:
        base_dir: the directory where images are stored
        filename: file with list of tile names
    Return:
        Means of each channel
    """

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    summation = numpy.zeros(3, dtype=numpy.float64)
    num_pixel = 0
    for tile in tiles:
        image = tifffile.imread(base_dir + tile + ".tif")
        image = image.astype(float)
        image = image / 65535
        num_pixel = image.shape[0] * image.shape[1]
        sum_image = numpy.sum(image, axis=(0,1))
        summation += sum_image

    summation = summation/(num_pixel * len(tiles))
    print("Mean")
    print(summation)
    return summation

def find_std_dev_hs(fname, base_dir, means):
    """
    Function to compute std dev of hs image
    Args:
        base_dir: the directory where images are stored
        filename: file with list of tile names
        means: list of the menas of each channel
    """

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    std_dev = numpy.zeros(3, dtype=numpy.float64)
    num_pixel = 0
    for tile in tiles:
        image = tifffile.imread(base_dir + tile + ".tif")
        image = image.astype(float)
        image = image / 65535
        num_pixel = image.shape[0] * image.shape[1]
        dev_image = numpy.subtract(image, means)
        var = numpy.square(dev_image)
        sum_var = numpy.sum(var, axis=(0,1))
        std_dev = std_dev + sum_var

    std_dev = std_dev/(num_pixel * len(tiles))
    std_dev = numpy.sqrt(std_dev)
    print("Std. Dev.")
    print(std_dev)

if __name__ == "__main__":
    #Input graph files
    
    fname = sys.argv[1]
    base_dir = sys.argv[2]
    img_type = sys.argv[3]
    if img_type == 'rgb':
        means = find_mean_rgb(fname,base_dir)
        std_devs = find_std_dev_rgb(fname, base_dir, means)
    if img_type == 'hs':
        means = find_mean_hs(fname,f'{base_dir}/123/')
        std_devs = find_std_dev_hs(fname, f'{base_dir}/123/', means)
        means = find_mean_hs(fname,f'{base_dir}/456/')
        std_devs = find_std_dev_hs(fname, f'{base_dir}/456/', means)
        means = find_mean_hs(fname,f'{base_dir}/78/')
        std_devs = find_std_dev_hs(fname, f'{base_dir}/78/', means)
