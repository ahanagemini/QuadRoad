import os
import sys
from scipy import misc
import numpy
import random

'''
Code to rotate input and ground truth images by random angle but ensure same angle for both for a tile
Args: rgb_dir=the directory where images are stored
      hght_dir=the directory where lidar images are stored
      hs_dir= base directory where hyperspectral images are saved
      filename= a list of the file names of the images for which computation is done
      target_dir: directory for ground truth
      save_dir: directory to save rotated images
      t_save_dir: directory to save rotated gt
      h_save_dir:directory to save rotated hght
'''
def save_rotated_images(fname, rgb_dir, target_dir, hght_dir,
        save_dir, t_save_dir, h_save_dir):

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    #summation = numpy.zeros(3, dtype=numpy.float64)
    for tile in tiles:
        image = misc.imread(rgb_dir + tile + ".png")
        hght = misc.imread(hght_dir + tile + ".png")
        target = misc.imread(target_dir + tile + ".png")
        rand_rot = random.randint(1,3)
        rot = rand_rot * 90
        image = misc.imrotate(image, rot)
        target = misc.imrotate(target, rot)
        print(rot)
        to_save = misc.toimage(image, cmin=0, cmax=255).save(save_dir+tile+"_r.png")
        to_save = misc.toimage(hght, cmin=0, cmax=255).save(h_save_dir+tile+"_r.png")
        to_save = misc.toimage(target, cmin=0, cmax=255).save(t_save_dir+tile+"_r.png")


if __name__ == "__main__":
    #Input graph files
    
    fname = sys.argv[1]
    rgb_dir = sys.argv[2]
    target_dir = sys.argv[3]
    save_dir = sys.argv[4]
    t_save_dir = sys.argv[5]
    hght_dir = sys.argv[6]
    h_save_dir = sys.argv[7]
    save_rotated_images(fname, rgb_dir, target_dir, hght_dir,
            save_dir, t_save_dir, h_save_dir)
