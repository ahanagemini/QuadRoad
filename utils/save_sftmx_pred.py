import os
import sys
from scipy import misc
import numpy
import random
from sklearn.cluster import KMeans
'''
Code to save the stfmx images after averaging over 2 to 4 sftmax values per pixel
      result directories are in decreasing order of weights
Args: res1_dir=the directory where first set of results are stored (max weight)
      filename= a list of the file names of the images for which computation is done
      res2_dir: directory where second set of  results are stored
      gt_dir: The ground truth directory
      save_dir: Directory where we save the softmax results
      percent: the threshold percent
      res3_dir: the directory where third set of results are stored, 'unused' indicates average of 2
      res4_dir_dir: the directory where result derived from 17 class is stored, unused indicates avg of <4 sets
      w1, w2, w3, w4: weights for each result
'''

def save_sftmx(percent, w1,w2,w3,w4):

    fname = "/home/ahana/road_data/valid.txt"
    res1_dir = "/home/ahana/road_data/exp/eval/tf_new/rgb_aug_ce/sftmx_results/"
    res2_dir = "/home/ahana/road_data/exp/eval/tf_new/hght_aug_ce/sftmx_results/"
    res3_dir = "/home/ahana/road_data/exp/eval/tf_new/hs_ce/sftmx_results/"
    res4_dir = "/home/ahana/road_data/exp/eval/tf_new/rgb_aug_17c_ce/sftmx_results1/"
    #res4_dir = "unused"
    save_dir = "/home/ahana/road_data/comb_results/"
    gt_dir = "/home/ahana/road_data/annotations/"

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    total_intersection = 0
    total_union = 0
    total_neg = 0
    total_diff = 0
    for tile in tiles:
        print(tile)
        res1 = misc.imread(res1_dir + tile + ".png")
        res2 = misc.imread(res2_dir + tile + ".png")
        if res4_dir != 'unused':
            print("Averaging 4")
            res4 = misc.imread(res4_dir + tile + ".png")
            res3 = misc.imread(res3_dir + tile + ".png")
            #hsp = 200 - hsp
            res1 = res1.astype(float)
            res2 = res2.astype(float)
            res3 = res3.astype(float)
            res4 = res4.astype(float)

            res3 = numpy.multiply(res3,w3)
            res1 = 200.0 - res1 #just a nuance due to having sometimes 0 as road and sometimes reverse
            res2 = 200.0 -res2
            res1 = numpy.multiply(res1,w1)
            res2 = numpy.multiply(res2,w2)
            #res4 = 200.0 - res4
            res4 = numpy.multiply(res4,w4)
        elif res3_dir != 'unused':
            print("Averaging 3")
            res3 = misc.imread(res3_dir + tile + ".png")
            res1 = res1.astype(float)
            res2 = res2.astype(float)
            res3 = res3.astype(float)

            #hsp = 200 - hsp
            res3 = numpy.multiply(res3,w3)
            res1 = 200.0 - res1 #just a nuance due to having sometimes 0 as road and sometimes reverse
            res2 = 200.0 -res2
            res1 = numpy.multiply(res1,w1)
            res2 = numpy.multiply(res2,w2)
        else:
            res1 = res1.astype(float)
            res2 = res2.astype(float)
            res1 = 200.0 - res1
            res2 = 200.0 - res2

            res1 = numpy.multiply(res1,w1)
            res2 = numpy.multiply(res2,w2)
        
        target = misc.imread(gt_dir + tile + ".png")
        if res4_dir != 'unused':
            combp_temp = numpy.add(res1,res2)
            combp_temp = numpy.add(combp_temp,res3)
            combp = numpy.add(combp_temp,res4)
        elif res3_dir != 'unused':
            combp_temp = numpy.add(res1,res2)
            combp = numpy.add(combp_temp,res3)
        else:
            combp = numpy.add(rgbp,hghtp)
        
        rev_combp = 200.0 - combp
        if save_dir != 'unused':
            misc.toimage(rev_combp, cmin=0, cmax=255).save(save_dir+"sftmx_results/"+tile+".png")
        
        #print(numpy.max(combp))
        #Thresholding to get predictions
        combp[combp < percent*2] = 1
        combp[combp >= percent*2] = 0
        #print(numpy.max(combp))
        #print(numpy.min(combp))
        diff_sum = numpy.count_nonzero(combp!=target)
        combp[combp==1] = 2
        intersection = numpy.count_nonzero(combp==target)
        combp[combp==2] = 1
        if save_dir != 'unused':
            misc.toimage(combp, cmin=0, cmax=255).save(save_dir+"pred_results/"+tile+".png")
        
        true_neg = numpy.count_nonzero(combp==target) - intersection
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
    
    #fname = sys.argv[1]
    #res1_dir = sys.argv[2]
    #save_dir = sys.argv[12]
    #gt_dir = sys.argv[7]
    #res2_dir = sys.argv[3]
    percent = float(sys.argv[1])
    #res3_dir = sys.argv[4]
    #res4_dir = sys.argv[5]
    w1 = float(sys.argv[2])
    w2 = float(sys.argv[3])
    w3 = float(sys.argv[4])
    w4 = float(sys.argv[5])
    #means = find_mean(fname,base_dir)
    #std_devs = find_std_dev(fname, base_dir, means)
    save_sftmx(percent, w1, w2, w3, w4)
