import os
import sys
from scipy import misc
import numpy
import random
from sklearn.cluster import KMeans
'''
Code to find iou of 17 classes
Args: base_dir=the directory where images are stored
      filename= a list of the file names of the images for which computation is done
      gt_dir: directory with ground truth for comparison
'''
def compute_metric(fname, base_dir, gt_dir):

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    total_intersection = [0]*17
    total_union = [0]*17
    total_neg = [0]*17
    total_diff = [0]*17
    total_gt = [0]*17
    spurious = [0]*17
    missing = [0]*17
    for tile in tiles:
        print(tile)
        pred = misc.imread(base_dir + tile + ".png")
        target = misc.imread(gt_dir + tile + ".png")
        pred_tmp = numpy.copy(pred)
        #pred = 1 -pred
        #target = 1-target
        for label in range(1,17):
            pred_tmp = numpy.copy(pred)
            pred_tmp[pred_tmp!=label] = 0 #only if pred == label, it is considered
            target_tmp = numpy.copy(target)
            target_tmp[target_tmp!=label] = 0 # only if target == label, it is considered
            diff_sum = numpy.count_nonzero(pred_tmp!=target_tmp) # if not equal set to lbel in one not in other
            pred_tmp[pred_tmp==0] = 17 # only equal now if both set a pixel to label
            intersection = numpy.count_nonzero(pred_tmp==target_tmp)
            pred_tmp[pred_tmp==17] = 0 #equal if both set as label or both do not
            true_neg = numpy.count_nonzero(pred_tmp==target_tmp) - intersection
            union = diff_sum + intersection
            total_intersection[label] += intersection
            total_union[label] += union
            total_neg[label] += true_neg
            total_diff[label] += diff_sum
            total_gt[label] += numpy.count_nonzero(target_tmp)

        pred_tmp = numpy.copy(pred)
        target_tmp = numpy.copy(target)
        target_tmp[pred!=1] = 17 # actual values only if pred is 1 (road)
        pred_tmp[target!=1] = 17 # actual values values only target is 1
        for label in range(0,17):
            spurious[label] += numpy.count_nonzero(target_tmp == label) #predicted as road but is actually label
            missing[label] += numpy.count_nonzero(pred_tmp == label) # predicted as other but os actually road
    for label in range(0,17):
        print("Label:"+str(label))
        print("Intersection:"+str(total_intersection[label]))
        print("Union:"+str(total_union[label]))
        print("Diff:"+str(total_diff[label]))
        print("True_neg:"+str(total_neg[label]))
        print("GT:"+str(total_gt[label]))
        print("Spurious road:"+str(spurious[label]))
        print("Missing road:"+str(missing[label]))
        if total_union[label] > 0:
            iou = total_intersection[label]/total_union[label]
        else:
            iou = 0.0
        print("IoU: "+ str(iou))


if __name__ == "__main__":
    #Input graph files
    
    fname = sys.argv[1]
    base_dir = sys.argv[2]
    #save_dir = sys.argv[4]
    gt_dir = sys.argv[3]
    #rgb_dir = sys.argv[5]
    #means = find_mean(fname,base_dir)
    #std_devs = find_std_dev(fname, base_dir, means)
    compute_metric(fname, base_dir, gt_dir)
