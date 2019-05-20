import os
import numpy
import sys
from scipy import misc

base_dir = sys.argv[1]
fname = sys.argv[2]
tiles = open(fname).read().split("\n")
tiles = [t for t in tiles if t!= ""]
for i,tile in enumerate(tiles):
    rgb = misc.imread(base_dir+"/rgb/" + tile + ".png")
    hght_pred = misc.imread(base_dir+"/save_error_map/" + tile + ".png")
    pred_17_4c = misc.imread(base_dir+"/pred_17_4c/" + tile + ".png")
    #pred_17_1_pred = misc.imread(base_dir+"//" + tile + ".png")
    rgb_pred = misc.imread(base_dir+"/rgb_aug_pred/" + tile + ".png")
    #hs_pred = misc.imread(base_dir+"/test_pred_hs/" + tile + ".png")
    annot = misc.imread(base_dir+"/test_annot/" + tile + ".png")
    annot_17 = misc.imread(base_dir+"/ground_truth_500/" + tile + ".png")
    #hght_pred = hght_pred*255
    rgb_pred = rgb_pred*255
    #hs_pred = hs_pred*255
    annot = annot*255
    annot_17 = annot_17 * 15
    pred_17_4c = pred_17_4c* 15
    annot_pad = numpy.zeros((512,512), dtype='uint8')
    annot_pad[6:506,6:506] = annot
    rgb_pad = numpy.zeros((512,512,3), dtype='uint8')
    rgb_pad[6:506,6:506,:] = rgb
    hght_pred_pad = numpy.zeros((512,512), dtype='uint8')
    hght_pred_pad[6:506,6:506] = hght_pred
    #print(annot.dtype)
    #pred_17_1_pred = pred_17_1_pred*255
    annot_3 = numpy.zeros((512,512,3), dtype='uint8')
    annot_3[:,:,0] = annot_pad
    annot_3[:,:,1] = annot_pad
    annot_3[:,:,2] = annot_pad
    hght_pred_3 = numpy.zeros((512,512,3), dtype='uint8')
    hght_pred_3[:,:,0] = hght_pred_pad
    hght_pred_3[:,:,1] = hght_pred_pad
    hght_pred_3[:,:,2] = hght_pred_pad
    rgb_pred_3 = numpy.zeros((512,512,3), dtype='uint8')
    rgb_pred_3[:,:,0] = rgb_pred
    rgb_pred_3[:,:,1] = rgb_pred
    rgb_pred_3[:,:,2] = rgb_pred
    #hs_pred_3 = numpy.zeros((512,512,3), dtype='uint8')
    #hs_pred_3[:,:,0] = hs_pred
    #hs_pred_3[:,:,1] = hs_pred
    #hs_pred_3[:,:,2] = hs_pred
    #pred_17_1_pred_3 = numpy.zeros((512,512,3), dtype='uint8')
    #pred_17_1_pred_3[:,:,0] = pred_17_1_pred
    #pred_17_1_pred_3[:,:,1] = pred_17_1_pred
    #pred_17_1_pred_3[:,:,2] = pred_17_1_pred
    pred_17_4c_pred_3 = numpy.zeros((512,512,3), dtype='uint8')
    pred_17_4c_pred_3[:,:,0] = pred_17_4c
    pred_17_4c_pred_3[:,:,1] = pred_17_4c
    pred_17_4c_pred_3[:,:,2] = pred_17_4c
    annot_17_pad = numpy.zeros((512,512), dtype='uint8')
    annot_17_pad[6:506,6:506] = annot_17
    annot_17_3 = numpy.zeros((512,512,3), dtype='uint8')
    annot_17_3[:,:,0] = annot_17_pad
    annot_17_3[:,:,1] = annot_17_pad
    annot_17_3[:,:,2] = annot_17_pad
    r1 = numpy.concatenate((rgb_pad,annot_3, pred_17_4c_pred_3),axis=1)
    r2 = numpy.concatenate((hght_pred_3,rgb_pred_3, annot_17_3),axis=1)
    print(r1.shape)
    print(r2.shape)
    to_save = numpy.concatenate((r1,r2), axis=0)
    print(to_save.shape)
    misc.toimage(to_save, cmin=0, cmax=255).save(base_dir+"/to_show/" +"tile_"+ str(240-i) +"_"+ tile + ".png")

