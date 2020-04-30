import os
import sys
from scipy import misc
import numpy
import random
from sklearn.cluster import KMeans
'''
Code to identify classes most often misclassified as road pixels
Args: base_dir=the directory where images are stored
      filename= a list of the file names of the images for which computation is done
      gt_dir: ground truth to compare with
'''
def identify_confusion(fname, base_dir, gt_dir):
    """
    Function to compute the number of pixels of each class
    that is incorrectly predicted as road class
    Args:
        base_dir: the directory where images are stored
        filename: file with list of tile names
        gt_dir: ground truth to compare with
    """
    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    sums = numpy.zeros(17)
    for tile in tiles:
        pred = misc.imread(base_dir + tile + ".png")
        target = misc.imread(gt_dir + tile + ".png")
        target[pred!=1] = 17 # so that only for road (label=1) can the two be equal
        for i in range(17):
            sums[i] += numpy.count_nonzero(target==i)

    for i in range(17):
        print(f'{i}: {sums[i]}')

if __name__ == "__main__":
    #Input graph files
    
    fname = sys.argv[1]
    base_dir = sys.argv[2]
    gt_dir = sys.argv[3]
    identify_confusion(fname, base_dir, gt_dir)
