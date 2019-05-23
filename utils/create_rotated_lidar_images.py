import os
import sys
from scipy import misc
import numpy
import random
'''
Code to find the rotatin angle for each tile target and rotate lidar by same amount
Args: lidar_dir=the directory where lidar images are stored
      filename= a list of the file names of the images for which computation is done
      target_dir: that has the target images
      save_dir: directory to save the images
'''
def save_rotated_images(fname, lidar_dir, target_dir, save_dir):

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    #summation = numpy.zeros(3, dtype=numpy.float64)
    for tile in tiles:
        image = misc.imread(lidar_dir + tile + ".png")
        target = misc.imread(target_dir + tile + ".png")
        rot_target = misc.imread(target_dir + tile + "_r.png")
        #image = image.astype(float)
        #image = image / 255
        #sum_image = numpy.sum(image, axis=(0,1))
        #summation += sum_image
        #sum_image = sum_image / 250000
        #std_dev = find_std_dev(image, sum_image)
        for i in range(1,4):
            rot = 90*i
            test_target = misc.imrotate(target,rot)
            if numpy.array_equal(test_target, rot_target):
                image = misc.imrotate(image, rot)
                print(rot)
                to_save = misc.toimage(image, cmin=0, cmax=255).save(save_dir+tile+"_r.png")

    #summation = summation/(250000 * len(tiles))
    #print(summation)

if __name__ == "__main__":
    #Input graph files
    
    fname = sys.argv[1]
    lidar_dir = sys.argv[2]
    target_dir = sys.argv[3]
    save_dir = sys.argv[4]
    #t_save_dir = sys.argv[5]
    #means = find_mean(fname,base_dir)
    #std_devs = find_std_dev(fname, base_dir, means)
    save_rotated_images(fname, lidar_dir, target_dir, save_dir)
