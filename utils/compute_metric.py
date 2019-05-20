import os
import sys
from scipy import misc
import numpy
import random
from sklearn.cluster import KMeans
'''
Code to find the mean and std. dev. of  number of images channel-wise
Args: base_dir=the directory where images are stored
      filename= a list of the file names of the images for which computation is done
'''
def compute_metric(fname, base_dir, gt_dir):

    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    total_intersection = 0
    total_union = 0
    total_neg = 0
    total_diff = 0
    #L=[]
    #error_pixels=[]
    #error_locs=[]
    #color=[(255,0,0),(128,0,0),(255,128,0),(255,255,0),(128,128,0),(128,255,0),(255,0,128),(128,0,128),(255,128,128),(255,255,128),(128,128,128),(128,255,128),(255,0,255),(128,0,255),(255,128,255),(255,255,255),(128,128,255),(128,255,255), (0,128,255), (0,255, 255)]
    #summation = numpy.zeros(3, dtype=numpy.float64)
    for tile in tiles:
        #print(tile)
        pred = misc.imread(base_dir + tile + ".png")
        target = misc.imread(gt_dir + tile + ".png")
        #rgb = misc.imread(rgb_dir + tile + ".png")
        #image = image / 25i5
        #image = image[6:506,6:506]
        #i,j = numpy.where(target!=image)
        #rgb_error = rgb[i,j]
        #print(rgb_error.shape)
        #print(rgb_error.size)
        #pred = 1 -pred
        #target = 1-target
        diff_sum = numpy.count_nonzero(pred!=target)
        pred[pred==1] = 2
        intersection = numpy.count_nonzero(pred==target)
        pred[pred==2] = 1
        true_neg = numpy.count_nonzero(pred==target) - intersection
        union = diff_sum + intersection
        total_intersection += intersection
        total_union += union
        total_neg += true_neg
        total_diff += diff_sum

    print(total_intersection)
    print(total_union)
    print(total_diff)
    print(total_neg)
    iou = total_intersection/total_union
    print(iou)
        #to_save = misc.toimage(target, cmin=0, cmax=255).save(save_dir+tile+".png")
    #summation = summation/(250000 * len(tiles))
    #print(summation)
    #sorted_L = sorted(L, key=lambda tup: tup[1])
    #for i in range(2400):
    #    print(sorted_L[i])

    #error_pixels_np = numpy.concatenate(error_pixels, axis=0)
    #kmeans = KMeans(n_clusters=20, random_state=0).fit(error_pixels_np)
    #for i,tile in enumerate(tiles):
    #    cluster = kmeans.predict(error_pixiels[i])
    #    print(cluster)
        #img=cv2.imread(rgb_dir + tile + ".png", cv2.IMREAD_GRAYSCALE)
        #for j in range(cluster.size):
        #    rgb_error=cv2.circle(img, error_locs[j], 2, color[cluster[j]], -1)
        #cv2.imwrite(save_dir+tile+".png",img)

def find_std_dev(image, means):

    std_dev = numpy.zeros(1, dtype=numpy.float64)
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
    #save_dir = sys.argv[4]
    gt_dir = sys.argv[3]
    #rgb_dir = sys.argv[5]
    #means = find_mean(fname,base_dir)
    #std_devs = find_std_dev(fname, base_dir, means)
    compute_metric(fname, base_dir, gt_dir)
