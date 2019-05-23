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
def compute_metric(fname, base_dir, gt_dir, road_label):

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    total_intersection = 0
    total_union = 0
    total_neg = 0
    total_diff = 0
    for tile in tiles:
        #print(tile)
        pred = misc.imread(base_dir + tile + ".png")
        target = misc.imread(gt_dir + tile + ".png")
        if road_label == 1:
            pred = 1 -pred
        #target = 1-target
        diff_sum = numpy.count_nonzero(pred!=target)
        pred[pred==1] = 2 # so that only for road (label=0) can the two be equal
        intersection = numpy.count_nonzero(pred==target)
        pred[pred==2] = 1
        true_neg = numpy.count_nonzero(pred==target) - intersection
        union = diff_sum + intersection
        total_intersection += intersection
        total_union += union
        total_neg += true_neg
        total_diff += diff_sum

    print(total_intersection)
    print(total_union)
    print(total_diff)
    print(total_neg)
    iou = total_intersection/total_union
    print(iou)

if __name__ == "__main__":
    #Input graph files
    
    fname = sys.argv[1]
    base_dir = sys.argv[2]
    road_label = sys.argv[4]
    gt_dir = sys.argv[3]
    compute_metric(fname, base_dir, gt_dir, road_label)
