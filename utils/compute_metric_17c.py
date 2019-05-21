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
    total_intersection = [0]*17
    total_union = [0]*17
    total_neg = [0]*17
    total_diff = [0]*17
    total_gt = [0]*17
    spurious = [0]*17
    missing = [0]*17
    #L=[]
    #error_pixels=[]
    #error_locs=[]
    #color=[(255,0,0),(128,0,0),(255,128,0),(255,255,0),(128,128,0),(128,255,0),(255,0,128),(128,0,128),(255,128,128),(255,255,128),(128,128,128),(128,255,128),(255,0,255),(128,0,255),(255,128,255),(255,255,255),(128,128,255),(128,255,255), (0,128,255), (0,255, 255)]
    #summation = numpy.zeros(3, dtype=numpy.float64)
    for tile in tiles:
        print(tile)
        pred = misc.imread(base_dir + tile + ".png")
        target = misc.imread(gt_dir + tile + ".png")
        pred_tmp = numpy.copy(pred)
        #rgb = misc.imread(rgb_dir + tile + ".png")
        #image = image / 25i5
        #image = image[6:506,6:506]
        #i,j = numpy.where(target!=image)
        #rgb_error = rgb[i,j]
        #print(rgb_error.shape)
        #print(rgb_error.size)
        #pred = 1 -pred
        #target = 1-target
        for label in range(1,17):
            pred_tmp = numpy.copy(pred)
            pred_tmp[pred_tmp!=label] = 0
            target_tmp = numpy.copy(target)
            target_tmp[target_tmp!=label] = 0
            diff_sum = numpy.count_nonzero(pred_tmp!=target_tmp)
            pred_tmp[pred_tmp==0] = 17
            intersection = numpy.count_nonzero(pred_tmp==target_tmp)
            pred_tmp[pred_tmp==17] = 0
            true_neg = numpy.count_nonzero(pred_tmp==target_tmp) - intersection
            union = diff_sum + intersection
            total_intersection[label] += intersection
            total_union[label] += union
            total_neg[label] += true_neg
            total_diff[label] += diff_sum
            total_gt[label] += numpy.count_nonzero(target_tmp)

        pred_tmp = numpy.copy(pred)
        target_tmp = numpy.copy(target)
        target_tmp[pred!=1] = 17
        pred_tmp[target!=1] = 17
        for label in range(0,17):
            spurious[label] += numpy.count_nonzero(target_tmp == label)
            missing[label] += numpy.count_nonzero(pred_tmp == label)
    for label in range(0,17):
        print("Label:"+str(label))
        print("Intersection:"+str(total_intersection[label]))
        print("Union:"+str(total_union[label]))
        print("Diff:"+str(total_diff[label]))
        print("True_neg:"+str(total_neg[label]))
        print("GT:"+str(total_gt[label]))
        print("Spurious road:"+str(spurious[label]))
        print("Missing road:"+str(missing[label]))
        if total_union[label] > 0:
            iou = total_intersection[label]/total_union[label]
        else:
            iou = 0.0
        print("IoU: "+ str(iou))
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
