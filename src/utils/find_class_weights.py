import sys
import os
import numpy
from scipy import misc

'''
Usage: find_class_weights.py path fname num_classes
Code to find the class weights for class balancing
Args: path: path wher ethe ground truths are saved
      fname: file with list of tiles
      num_class: number of classes
'''
def find_class_weights(path, fname, num_class):
    """
    Function to find the weighr for each class
    Uses median frequency balancing
    Args:
        path: path wher ethe ground truths are saved
        fname: file with list of tiles
        num_class: number of classes
    """
    total_counts = numpy.zeros(num_class)
    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    for filename in tiles:
        image = misc.imread(path+filename+'.png')
        flat_image = image.reshape(-1)  # makes one long line of pixels
        labels, counts = numpy.unique(flat_image, return_counts = True, axis = 0)
        for i,label in enumerate(labels):
            total_counts[label] = total_counts[label] + counts[i]
 
    total_counts_list = total_counts.tolist()
    total_counts_list.sort(key=float)
    if num_class % 2 > 0:
        median = total_counts_list[num_class // 2]
    else:
        median =  total_counts_list[num_class // 2] + total_counts_list[num_class // 2 - 1]
    print(median)
    total_counts = 1/total_counts
    total_counts = total_counts*median
    print(total_counts)

if __name__=="__main__":
    path = sys.argv[1]
    fname = sys.argv[2]
    num_class = int(sys.argv[3])
    find_class_weights(path, fname, num_class)
