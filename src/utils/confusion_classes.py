import os
import sys
from scipy import misc
import numpy
import random
from sklearn.cluster import KMeans
'''
Code to compute iou for 2 class classification images for road only
Args: base_dir=the directory where images are stored
      filename= a list of the file names of the images for which computation is done
      gt_dir: ground truth to compare with
      road_label: label for road, 0 or 1
'''
def compute_metric(fname, base_dir, gt_dir):

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    total_intersection = 0
    total_union = 0
    total_neg = 0
    total_diff = 0
    total_fp = 0
    total_fn = 0
    sums = numpy.zeros(17)
    for tile in tiles:
        #print(tile)
        pred = misc.imread(base_dir + tile + ".png")
        target = misc.imread(gt_dir + tile + ".png")
        #pred[pred < 100] = 0
        #pred[pred >= 100] = 1
        #target = 1-target
        target[pred!=1] = 17 # so that only for road (label=0) can the two be equal
        for i in range(17):
            sums[i] += numpy.count_nonzero(target==i)

    for i in range(17):
        print(f'{i}: {sums[i]}')
if __name__ == "__main__":
    #Input graph files
    
    fname = sys.argv[1]
    base_dir = sys.argv[2]
    gt_dir = sys.argv[3]
    compute_metric(fname, base_dir, gt_dir)
