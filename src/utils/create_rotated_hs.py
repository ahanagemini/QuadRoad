import os
import sys
from scipy import misc
import numpy
import random
import cv2


'''
Code to find the rotatin angle for each tile target and rotate hs by same amount
Args: hs_dir=the directory where lidar images are stored
      filename= a list of the file names of the images for which computation is done
      target_dir: that has the target images
      save_dir: directory to save the images
'''

def save_rotated_hs(fname, hs_dir, target_dir, save_dir):

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    for tile in tiles:
        image123 = cv2.imread(hs_dir + '123/' + tile + ".tif", -1)
        image456 =  cv2.imread(hs_dir + '456/' + tile + ".tif", -1)
        image78 = cv2.imread(hs_dir + '78/' + tile + ".tif", -1)
        target = cv2.imread(target_dir + tile + ".png", cv2.IMREAD_UNCHANGED)
        rot_target = cv2.imread(target_dir + tile + "_r.png",cv2.IMREAD_UNCHANGED)
        print(rot_target.shape)
        print(target.shape)
        print(tile)
        for i in range(1,4):
            test_target = numpy.rot90(target,i)
            if numpy.array_equal(test_target, rot_target):
                dst123 = numpy.rot90(image123 ,i)
                dst456 = numpy.rot90(image456 ,i)
                dst78 = numpy.rot90(image78 ,i)
                print(90*i)
                cv2.imwrite(save_dir  + '123/'+tile+"_r.tif", dst123)
                cv2.imwrite(save_dir  + '456/'+tile+"_r.tif", dst456)
                cv2.imwrite(save_dir  + '78/'+tile+"_r.tif", dst78)

if __name__ == "__main__":
    
    fname = sys.argv[1]
    hs_dir = sys.argv[2]
    target_dir = sys.argv[3]
    save_dir = sys.argv[4]
    save_rotated_hs(fname, hs_dir, target_dir, save_dir)
