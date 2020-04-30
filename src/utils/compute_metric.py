import os
import sys
from scipy import misc
import numpy
import random
from sklearn.cluster import KMeans
'''
Code to find iou of all classes
Args: base_dir=the directory where images are stored
      filename= a list of the file names of the images for which computation is done
      gt_dir: directory with ground truth for comparison
      num_class: number of classes
'''
def compute_metric(fname, base_dir, gt_dir, num_class):
    """
    Function to compute the IOU and other metrics
    for predictions
    Args:
        base_dir: the directory where predictions are stored
        fname: a file with list of tile names
        gt_dir: directory with ground truth for comparison
        num_class: number of classes
    """
    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    total_intersection = [0] * num_class
    total_union = [0] * num_class
    total_neg = [0]*num_class
    total_diff = [0]*num_class
    total_gt = [0]*num_class
    spurious = [0]*num_class
    missing = [0]*num_class
    for tile in tiles:
        print(tile)
        pred = misc.imread(base_dir + tile + ".png")
        target = misc.imread(gt_dir + tile + ".png")
        pred_tmp = numpy.copy(pred)
        #pred = 1 -pred
        #target = 1-target
        for label in range(1,num_class):
            pred_tmp = numpy.copy(pred)
            pred_tmp[pred_tmp!=label] = 0 #only if pred == label, it is considered
            target_tmp = numpy.copy(target)
            target_tmp[target_tmp!=label] = 0 # only if target == label, it is considered
            diff_sum = numpy.count_nonzero(pred_tmp!=target_tmp) # if not equal set to lbel in one not in other
            pred_tmp[pred_tmp==0] = num_class # only equal now if both set a pixel to label
            intersection = numpy.count_nonzero(pred_tmp==target_tmp)
            pred_tmp[pred_tmp==num_class] = 0 #equal if both set as label or both do not
            true_neg = numpy.count_nonzero(pred_tmp==target_tmp) - intersection
            union = diff_sum + intersection
            total_intersection[label] += intersection
            total_union[label] += union
            total_neg[label] += true_neg
            total_diff[label] += diff_sum
            total_gt[label] += numpy.count_nonzero(target_tmp)

        pred_tmp = numpy.copy(pred)
        target_tmp = numpy.copy(target)
        target_tmp[pred!=1] = num_class # actual values only if pred is 1 (road)
        pred_tmp[target!=1] = num_class # actual values values only target is 1
        for label in range(0,num_class):
            spurious[label] += numpy.count_nonzero(target_tmp == label) #predicted as road but is actually label
            missing[label] += numpy.count_nonzero(pred_tmp == label) # predicted as other but os actually road
    for label in range(0,num_class):
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
    gt_dir = sys.argv[3]
    num_class = int(sys.argv[4])
    compute_metric(fname, base_dir, gt_dir, num_class)
