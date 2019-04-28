from scipy import misc
import numpy
import sys
import os

def combine_preds(rgb_dir,hght_dir,hs_dir, file_name):

    f = open(file_name, "r")
    lines = f.read().splitlines()

    for tile in lines:
        print(tile)
        rgb = misc.imread(rgb_dir+tile+'.png')
        hght = misc.imread(hght_dir+tile+'.png')
        hs = misc.imread(hs_dir+tile+'.png')
        to_save = numpy.zeros((500, 500, 3))
        to_save[:,:,0] = rgb[6:506,6:506]
        to_save[:,:,1] = hght[6:506,6:506]
        to_save[:,:,2] = hs[6:506,6:506]
        #to_save = to_save*255
        outFilepath = "/home/ahana/road_data/pred_3_heat_c/"+tile+'.png'
        misc.toimage(to_save, cmin=0, cmax=255).save(outFilepath)

if __name__=="__main__":
    rgb_dir = sys.argv[1]
    hght_dir = sys.argv[2]
    hs_dir = sys.argv[3]
    file_name = sys.argv[4]
    print("hi")
    print(file_name)
    combine_preds(rgb_dir,hght_dir,hs_dir,file_name)
