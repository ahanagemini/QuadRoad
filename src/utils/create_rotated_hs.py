import os
import sys
from scipy import misc
import numpy
import random
import cv2


'''
Code to find the rotatin angle for each tile target and rotate lidar by same amount
Args: lidar_dir=the directory where lidar images are stored
      filename= a list of the file names of the images for which computation is done
      target_dir: that has the target images
      save_dir: directory to save the images
'''

def save_rotated_images(fname, rgb_dir, target_dir, save_dir):

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    #summation = numpy.zeros(3, dtype=numpy.float64)
    for tile in tiles:
        image = cv2.imread(rgb_dir + tile + ".tif", -1)
        target = cv2.imread(target_dir + tile + ".png", cv2.IMREAD_UNCHANGED)
        rot_target = cv2.imread(target_dir + tile + "_r.png",cv2.IMREAD_UNCHANGED)
        #target = misc.imread(target_dir + tile + ".png")
        #rot_target = misc.imread(target_dir + tile + "_r.png")
        print(tile)
        #image = image.astype(float)
        #image = image / 255
        #sum_image = numpy.sum(image, axis=(0,1))
        #summation += sum_image
        #sum_image = sum_image / 250000
        #std_dev = find_std_dev(image, sum_image)
        for i in range(1,4):
            #rot = 90*i
            #print("check:"+str(90*i))
            test_target = numpy.rot90(target,i)
            #rows,cols = target.shape
            #M1 = cv2.getRotationMatrix2D((cols/2,rows/2),rot,1)
            #test_target = cv2.warpAffine(target,M1,(cols,rows))
            #test_target = misc.imrotate(target,rot)
            if numpy.array_equal(test_target, rot_target):
                #image = misc.imrotate(image, rot)
                #rows,cols,channel = image.shape
                #M = cv2.getRotationMatrix2D((cols/2,rows/2),rot,1)
                #dst = cv2.warpAffine(image,M,(cols,rows))
                dst = numpy.rot90(image,i)
                print(90*i)
                #to_save = misc.toimage(image, cmin=0, cmax=255).save(save_dir+tile+"_r.png")
                print(dst.shape)
                print(dst.dtype)
                cv2.imwrite(save_dir+tile+"_r.tif",dst)
    #summation = summation/(250000 * len(tiles))
    #print(summation)

def find_std_dev(image, means):

    std_dev = numpy.zeros(3, dtype=numpy.float64)
    dev_image = numpy.subtract(image, means)
    var = numpy.square(dev_image)
    sum_var = numpy.sum(var, axis=(0,1))
    avg_var = sum_var / 250000
    std_dev = numpy.sqrt(avg_var)
    return std_dev    

if __name__ == "__main__":
    #Input graph files
    
    fname = sys.argv[1]
    rgb_dir = sys.argv[2]
    target_dir = sys.argv[3]
    save_dir = sys.argv[4]
    #t_save_dir = sys.argv[5]
    #means = find_mean(fname,base_dir)
    #std_devs = find_std_dev(fname, base_dir, means)
    save_rotated_images(fname, rgb_dir, target_dir, save_dir)
