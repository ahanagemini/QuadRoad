import os
import sys
from scipy import misc
import numpy

def find_mean(fname, base_dir):

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    summation = numpy.zeros(1, dtype=numpy.float64)
    for tile in tiles:
        image = misc.imread(base_dir + tile + ".png")
        image = image.astype(float)
        image = image / 255
        sum_image = numpy.sum(image, axis=(0,1))
        summation += sum_image
    
    summation = summation/(250000 * len(tiles))
    print(summation)
    return summation

def find_std_dev(fname, base_dir, means):

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    std_dev = numpy.zeros(1, dtype=numpy.float64)
    for tile in tiles:
        image = misc.imread(base_dir + tile + ".png")
        image = image.astype(float)
        image = image / 255
        dev_image = numpy.subtract(image, means)
        var = numpy.square(dev_image)
        sum_var = numpy.sum(var, axis=(0,1))
        std_dev = std_dev + sum_var
        
    std_dev = std_dev/(250000 * len(tiles))
    print(std_dev)    

if __name__ == "__main__":
    #Input graph files
    
    fname = sys.argv[1]
    base_dir = sys.argv[2]
    means = find_mean(fname,base_dir)
    std_devs = find_std_dev(fname, base_dir, means)