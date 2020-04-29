import sys
import os
import numpy
from scipy import misc

'''
Code to find the class weights for class balancing for 2 class only
Args: path
'''
def find_class_weights(path, fname):
    total_counts = numpy.zeros(2)
    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    for filename in tiles:
        print(filename)
        image = misc.imread(path+filename+'.png')
        flat_image = image.reshape(-1)  # makes one long line of pixels
        #labels, counts = numpy.unique(flat_image, return_counts = True, axis = 0)
        #print(labels)
        #print(counts)
        #for i,label in enumerate(labels):
        #    total_counts[label] = total_counts[label] + counts[i]
        count = numpy.count_nonzero(image)
        total_counts[0] = total_counts[0] + (500*500 - count)
        total_counts[1] = total_counts[1] + count
    #total_counts_list = total_counts.tolist()
    #print(total_counts_list)
    #total_counts_list.sort(key=float)
    #print(total_counts_list)
     
    median = (total_counts[0] + total_counts[1])/2
    print(total_counts)
    print(median)
    total_counts = 1/total_counts
    total_counts = total_counts*median
    print(total_counts)

if __name__=="__main__":
    path = sys.argv[1]
    fname = sys.argv[2]
    find_class_weights(path, fname)
