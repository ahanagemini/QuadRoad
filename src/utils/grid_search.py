import os
import sys
from scipy import misc
import numpy
import random
from numpy.linalg import norm
from scipy.optimize import differential_evolution
import time
'''
Code to use differential evolution to genetically search a search space for 'n'
weights and a threshold percentage by using validation set,
find the best result and then use the best weights to compute accuracy on the test set
Args:
    num_params: the number of parameters
    paths_file: The file for the test set
Each line of the paths_file should have the following the the given sequence
Put unused if the parameter is not used
    fname= a list of the file names of the images for which computation is done
    gt_dir: The ground truth directory
    res1_dir=the directory where first set of results are stored (max weight)
    res2_dir: directory where second set of  results are stored
    res3_dir: the directory where third set of results are stored, 'unused' indicates average of 2
    res4_dir_dir: the directory where result derived from 17 class is stored, unused indicates avg of <4 sets
    save_dir: Directory where we save the softmax results

'''
def compute_metric(params, path_file="paths.txt"):

    '''
      Code to save the stfmx images after averaging over 2 to 4 sftmax values per pixel
      result directories are in decreasing order of weights
      Args: 
          params: the weights guessed for 2 to 4 models and the threshold percent
          paths_file: hardcoded for the validation set, use a separate one for the test set
    '''
    
    #fname = "/home/ahana/road_data/valid.txt"
    paths = open(path_file).read().split("\n")
    paths = [t for t in paths if t!= ""]
    fname = paths[0]
    gt_dir = paths[1]
    res1_dir = paths[2]
    res2_dir = paths[3]
    res3_dir = paths[4]
    res4_dir = paths[5]
    save_dir = paths[6]
    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    total_intersection = 0
    total_union = 0
    total_neg = 0
    total_diff = 0
    total_fp = 0
    total_fn = 0
    print(params)
    #assert n_members == len(params) -1
    for tile in tiles:
        # Compute the segmentation with current wts and threshold
        res1 = misc.imread(res1_dir + tile + ".png")
        res2 = misc.imread(res2_dir + tile + ".png")
        if len(params) == 5:
            #print("Averaging 4")
            res4 = misc.imread(res4_dir + tile + ".png")
            res3 = misc.imread(res3_dir + tile + ".png")
            #hsp = 200 - hsp
            res1 = res1.astype(float)
            res2 = res2.astype(float)
            res3 = res3.astype(float)
            res4 = res4.astype(float)
            res3 = numpy.multiply(res3,params[2])
            #res1 = 200.0 - res1 #just a nuance due to having sometimes 0 as road and sometimes reverse
            #res2 = 200.0 - res2
            res1 = numpy.multiply(res1,params[0])
            res2 = numpy.multiply(res2,params[1])
            #res4 = 200 - res4
            res4 = numpy.multiply(res4,params[3])
        elif len(params) == 4:
            #print("Averaging 3")
            res3 = misc.imread(res3_dir + tile + ".png")
            #hsp = 200 - hsp
            res1 = res1.astype(float)
            res2 = res2.astype(float)
            res3 = res3.astype(float)
            res3 = numpy.multiply(res3,params[2])
            #res1 = 200.0 - res1 #just a nuance due to having sometimes 0 as road and sometimes reverse
            #res2 = 200.0 -res2
            res1 = numpy.multiply(res1, params[0])
            res2 = numpy.multiply(res2,params[1])
        else:
            res1 = res1.astype(float)
            res2 = res2.astype(float)
            #res1 = 200.0 - res1
            #res2 = 200.0 - res2
            res1 = numpy.multiply(res1,params[0])
            res2 = numpy.multiply(res2,params[1])
        # Read the target image for comparing
        target = misc.imread(gt_dir + tile + ".png")
        if len(params) == 5:
            combp_temp = numpy.add(res1,res2)
            combp_temp = numpy.add(combp_temp,res3)
            combp = numpy.add(combp_temp,res4)
        elif len(params) == 4:
            combp_temp = numpy.add(res1,res2)
            combp = numpy.add(combp_temp,res3)
        else:
            combp = numpy.add(res1,res2)
        
        # save the new softmax values
        rev_combp = 200 - combp
        if save_dir != 'unused':
            misc.toimage(rev_combp, cmin=0, cmax=255).save(save_dir+"sftmx_results/"+tile+".png")

        #Thresholding to get predictions
        combp[combp < params[-1]*2] = 1
        combp[combp >= params[-1]*2] = 0
        diff_sum = numpy.count_nonzero(combp!=target)
        combp[combp==1] = 2
        intersection = numpy.count_nonzero(combp==target)
        fp = numpy.count_nonzero(combp==0) - intersection
        fn = diff_sum - fp
        
        # Save new predictions
        combp[combp==2] = 1
        if save_dir != 'unused':
            misc.toimage(combp, cmin=0, cmax=255).save(save_dir+"pred_results/"+tile+".png")
        # Compute the iou
        true_neg = numpy.count_nonzero(combp==target) - intersection
        union = diff_sum + intersection
        total_intersection += intersection
        total_union += union
        total_neg += true_neg
        total_diff += diff_sum
        total_fp += fp
        total_fn += fn
    iou = total_intersection/total_union
    tpr = total_intersection/(total_intersection + total_fn)
    tnr = total_neg / (total_neg + total_fp)
    return iou

def normalize(weights):
	# calculate l1 vector norm
	result = norm(weights, 1)
	# check for a vector of all zeros
	if result == 0.0:
		return weights
	# return normalized vector (unit norm)
	return weights / result

# loss function for optimization process, designed to be minimized
def loss_function(opt_params):
	# normalize weights
        percent = float(opt_params[-1])
        weights = opt_params[:-1]
        normalized = normalize(weights)
        params = list(normalized)
        #append the threshold percent
        params.append(percent)
	# calculate error rate
        return 1.0 - compute_metric(params)

if __name__=="__main__":
 
    n_members = int(sys.argv[1])
    paths_file = sys.argv[2]
    # bounds for the grid search
    bound_w = [(0.0, 1.0)  for _ in range(n_members)]
    bound_w.append((40,60))
    # global optimization of ensemble weights
    start = time.time()
    result = differential_evolution(loss_function, bound_w, maxiter=25, tol=1e-5)
    # get the chosen weights
    end = time.time()
    params = result['x']
    weights = normalize(params[:-1])
    percent = float(params[-1])
    print('Optimized Weights: %s' % weights)
    # evaluate chosen weights on validation and test set
    params_final = list(weights)
    params_final.append(percent)
    score = compute_metric(params_final)
    print('Optimized Weights Score: %.3f' % score)
    score = compute_metric(params_final, paths_file)
    print('Optimized Weights Score: %.3f' % score)
    print("Time taken: "+str(end - start))


