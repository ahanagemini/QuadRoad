from scipy import misc
import numpy
import sys
import os

'''
Code to combine different preds as different channels and create rgb image
    args: file_nae: path to file that has tile names
          rgb_dir: Directory with rgb softmax
          hght_dir: directory with hght softmax
          hs_dir: Directory with hs softmax or with 17 class predicted classes
          save_dir: Directory to save the images
          hs_or_17c: 1 if we are using 17 class preds and 2 for hyperspectral
                     0 if neither and we are using 3 classes softmax from 17 class
'''

def combine_preds(file_name,rgb_dir,hght_dir,hs_dir, save_dir, hs_or_17c):

    f = open(file_name, "r")
    lines = f.read().splitlines()

    for tile in lines:
        print(tile)
        rgb = misc.imread(rgb_dir+tile+'.png')
        hght = misc.imread(hght_dir+tile+'.png')
        hs = misc.imread(hs_dir+tile+'.png')
        if hs_or_17c == 1:
            hs[hs==16]=3
            hs = hs*16
        elif hs_or_17c == 2:
            hs = 200 - hs
        to_save = numpy.zeros((500, 500, 3), dtype=numpy.uint8)
        to_save[:,:,0] = rgb[0:500,0:500]
        to_save[:,:,1] = hght[0:500,0:500]
        to_save[:,:,2] = hs[0:500,0:500]
        #to_save = to_save*255
        outFilepath = save_dir+tile+'.png'
        misc.toimage(to_save, cmin=0, cmax=255).save(outFilepath)

if __name__=="__main__":
    rgb_dir = sys.argv[2]
    hght_dir = sys.argv[3]
    hs_dir = sys.argv[4]
    file_name = sys.argv[1]
    save_dir = sys.argv[5]
    hs_or_17c = int(sys.argv[6])
    print(file_name)
    combine_preds(file_name, rgb_dir,hght_dir,hs_dir,save_dir, hs_or_17c)
