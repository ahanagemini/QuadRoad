import cv2
import numpy as np
import os
import glob
from collections import defaultdict
import re


def get_fnames_to_merge():
    # Format: 000931_image_46680_5_15.png
    rgb_fnames = list(glob.glob("/home/ahana/road_data/rgb_augment/*.png"))

    # Format: 46680_5_15.png
    hght_pred_fnames = list(glob.glob("/home/ahana/road_data/hght_aug_pred/*.png"))

    # Format: 46680_5_15.png
    rgb_pred_fnames = list(glob.glob("/home/ahana/road_data/rgb_aug_pred/*.png"))

    #tile_nums = [v[:-4] for v in rgb_pred_fnames]
    print(len(rgb_fnames), len(hght_pred_fnames), len(rgb_pred_fnames))
#    assert len(rgb_fnames) == len(hght_pred_fnames) == len(rgb_pred_fnames)

    hold = defaultdict(list)

    # rgb
    patc = re.compile(r"image_(.*)\.png")
    for fn in rgb_fnames:
        k = os.path.basename(fn)[:-4]
        hold[k].append(fn)

    # rgb pred
    for fn in rgb_pred_fnames:
        k = os.path.basename(fn)[:-4]
        hold[k].append(fn)

    # hght pred
    for fn in hght_pred_fnames:
        k = os.path.basename(fn)[:-4]
        hold[k].append(fn)

    #for k in list(hold.keys())[:4]:
    #    print(hold[k])

    return hold

def merge_output_predictions():
    out_dir = "/home/ahana/road_data/pred_aug_gray"
    merge_fnames = get_fnames_to_merge()
#    print(merge_fnames)
#    filtered = "49922_"
    for tile_num in merge_fnames:
        print(tile_num)
#        if filtered not in tile_num:
#            continue
        rgb_f, rgb_pred_f, hght_pred_f = merge_fnames[tile_num]
        #blank = np.zeros((500,500), dtype='uint8')
        # Load all images as grayscale.
        # Enforce values to be between 0 and 255.
        gray = cv2.imread(rgb_f, 0)
        rgb_p = cv2.imread(rgb_pred_f, 0) * 255
        rgb_p_up = rgb_p[6:506,6:506]
        hght_p = cv2.imread(hght_pred_f, 0) * 255
        hght_p_up = hght_p[6:506,6:506]
        print(gray.shape, rgb_p_up.shape, hght_p_up.shape)

        # Combine the gray channels to get an image
        #comb = cv2.merge((gray, rgb_p, hght_p))
        comb = cv2.merge((gray, rgb_p_up, hght_p_up))
        # Save the resulting 3-channeled image
        out_fn = os.path.join(out_dir, tile_num + ".png")
        print(out_fn)
        cv2.imwrite(out_fn, comb)


if __name__ == "__main__":
    merge_output_predictions()

