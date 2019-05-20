import os
import sys
from scipy import misc
import numpy
import random
import cv2
'''
Code to find the mean and std. dev. of  number of images channel-wise
Args: base_dir=the directory where images are stored
      filename= a list of the file names of the images for which computation is done
'''
def save_shifted_image(fname, base_dir, save_dir):

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    #summation = numpy.zeros(3, dtype=numpy.float64)
    for tile in tiles:
        image123 = cv2.imread(base_dir+"123/" + tile + ".tif", -1)
        #image123 = image123.astype(float)
        image456 = cv2.imread(base_dir+"456/" + tile + ".tif", -1)
        #image456 = image456.astype(float)
        image78 = cv2.imread(base_dir+"78/" + tile + ".tif", -1)
        #image78 = image78.astype(float)
        cv2.imwrite(save_dir+"123/"+tile+".tif",image123)
        cv2.imwrite(save_dir+"456/"+tile+".tif",image456)
        cv2.imwrite(save_dir+"78/"+tile+".tif",image78)
        #image = image / 255
        print(image123.dtype)
        image123 = image123.astype(float)
        image456 = image456.astype(float)
        image78 = image78.astype(float)
        sum_image123 = numpy.sum(image123, axis=(0,1))
        #summation += sum_image
        sum_image123 = sum_image123 / 250000
        std_dev123 = find_std_dev(image123, sum_image123)
        sum_image456 = numpy.sum(image456, axis=(0,1))
        #summation += sum_image
        sum_image456 = sum_image456 / 250000
        std_dev456 = find_std_dev(image456, sum_image456)
        sum_image78 = numpy.sum(image78, axis=(0,1))
        #summation += sum_image
        sum_image78 = sum_image78 / 250000
        std_dev78 = find_std_dev(image78, sum_image78)
        rand_shift = random.randint(-100, 100)
        shift = rand_shift/1000
        print(shift)
        print(shift*std_dev123)
        image123 = image123 + shift*std_dev123
        image123 = image123.astype('uint16')
        #to_save = misc.toimage(image, cmin=0, cmax=255).save(save_dir+tile+"_s.png")
        cv2.imwrite(save_dir+"123/"+tile+"_s.tif",image123)
        image456 = image456 + shift*std_dev456
        image456 = image456.astype('uint16')
        #to_save = misc.toimage(image, cmin=0, cmax=255).save(save_dir+tile+"_s.png")
        cv2.imwrite(save_dir+"456/"+tile+"_s.tif",image456)
        image78 = image78 + shift*std_dev78
        #to_save = misc.toimage(image, cmin=0, cmax=255).save(save_dir+tile+"_s.png")
        image78 = image78.astype('uint16')
        cv2.imwrite(save_dir+"78/"+tile+"_s.tif",image78)
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
    base_dir = sys.argv[2]
    save_dir = sys.argv[3]
    #means = find_mean(fname,base_dir)
    #std_devs = find_std_dev(fname, base_dir, means)
    save_shifted_image(fname, base_dir, save_dir)
