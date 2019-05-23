import os
import sys
from scipy import misc
import numpy
import random
'''
Code to shift the intensity of each channel of rgb image by x(=[-0.1....0.1])*std_dev where x is randomly selected  
Args: base_dir=the directory where images are stored
      filename= a list of the file names of the images for which computation is done
      save_dir: directory to save the augmented images
'''
def save_shifted_image(fname, base_dir, save_dir):

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    #summation = numpy.zeros(3, dtype=numpy.float64)
    for tile in tiles:
        image = misc.imread(base_dir + tile + ".png")
        image = image.astype(float)
        #image = image / 255
        sum_image = numpy.sum(image, axis=(0,1))
        #summation += sum_image
        sum_image = sum_image / 250000
        std_dev = find_std_dev(image, sum_image)
        rand_shift = random.randint(-100, 100)
        shift = rand_shift/1000
        print(shift)
        print(shift*std_dev)
        image = image + shift*std_dev
        to_save = misc.toimage(image, cmin=0, cmax=255).save(save_dir+tile+"_s.png")
    #summation = summation/(250000 * len(tiles))
    #print(summation)

def find_std_dev(image, means):

    std_dev = numpy.zeros(1, dtype=numpy.float64)
    dev_image = numpy.subtract(image, means)
    var = numpy.square(dev_image)
    sum_var = numpy.sum(var, axis=(0,1))
    avg_var = sum_var / 250000
    std_dev = numpy.sqrt(avg_var)
    return std_dev    

if __name__ == "__main__":
    #Input graph files
    
    fname = sys.argv[1]
    base_dir = sys.argv[2]
    save_dir = sys.argv[3]
    #means = find_mean(fname,base_dir)
    #std_devs = find_std_dev(fname, base_dir, means)
    save_shifted_image(fname, base_dir, save_dir)
