import os
import sys
from scipy import misc
import numpy
import random
'''
Code to rotate input and ground truth images by random angle but ensure same angle for both for a tile
Args: rgb_dir=the directory where images are stored
      filename= a list of the file names of the images for which computation is done
      target_dir: directory for ground truth
      save_dir: directory to save rotated rgb
      t_save_dir: directory to save rotated target
'''
def save_rotated_images(fname, rgb_dir, target_dir, save_dir, t_save_dir):

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    #summation = numpy.zeros(3, dtype=numpy.float64)
    for tile in tiles:
        image = misc.imread(rgb_dir + tile + ".png")
        target = misc.imread(target_dir + tile + ".png")
        #image = image.astype(float)
        #image = image / 255
        #sum_image = numpy.sum(image, axis=(0,1))
        #summation += sum_image
        #sum_image = sum_image / 250000
        #std_dev = find_std_dev(image, sum_image)
        rand_rot = random.randint(1,3)
        rot = rand_rot * 90
        image = misc.imrotate(image, rot)
        target = misc.imrotate(target, rot)
        print(rot)
        #print(shift*std_dev)
        #image = image + shift*std_dev
        to_save = misc.toimage(image, cmin=0, cmax=255).save(save_dir+tile+"_r.png")
        to_save = misc.toimage(target, cmin=0, cmax=255).save(t_save_dir+tile+"_r.png")
    #summation = summation/(250000 * len(tiles))
    #print(summation)


if __name__ == "__main__":
    #Input graph files
    
    fname = sys.argv[1]
    rgb_dir = sys.argv[2]
    target_dir = sys.argv[3]
    save_dir = sys.argv[4]
    t_save_dir = sys.argv[5]
    #means = find_mean(fname,base_dir)
    #std_devs = find_std_dev(fname, base_dir, means)
    save_rotated_images(fname, rgb_dir, target_dir, save_dir, t_save_dir)
