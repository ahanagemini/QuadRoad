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
'''
def compute_metric(params, fname="/home/ahana/road_data/valid.txt"):

    '''
      Code to save the stfmx images after averaging over 2 to 4 sftmax values per pixel
      result directories are in decreasing order of weights
      Args: res1_dir=the directory where first set of results are stored (max weight)
          filename= a list of the file names of the images for which computation is done
          res2_dir: directory where second set of  results are stored
          gt_dir: The ground truth directory
          save_dir: Directory where we save the softmax results
          res3_dir: the directory where third set of results are stored, 'unused' indicates average of 2
          res4_dir_dir: the directory where result derived from 17 class is stored, unused indicates avg of <4 sets
          params: the weights guessed for 2 to 4 models and the threshold percent
    '''
    
    #fname = "/home/ahana/road_data/valid.txt"
    res1_dir = "/home/ahana/road_data/exp/eval/tf_new/rgb_aug_ce/sftmx_results/"
    res2_dir = "/home/ahana/road_data/exp/eval/tf_new/hght_aug_ce/sftmx_results/"
    res3_dir = "/home/ahana/road_data/exp/eval/tf_new/hs_ce/sftmx_results/"
    res4_dir = "/home/ahana/road_data/exp/eval/tf_new/rgb_aug_17c_ce/sftmx_results1/"
    #res4_dir = "unused"
    save_dir = 'unused'
    gt_dir = "/home/ahana/road_data/annotations/"
    tiles = open(fname).read().split("\n")
    tiles = [t for t in tiles if t!= ""]
    total_intersection = 0
    total_union = 0
    total_neg = 0
    total_diff = 0
    print(params)
    #assert n_members == len(params) -1
    for tile in tiles:
        #print(tile)
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
            res1 = 200.0 - res1 #just a nuance due to having sometimes 0 as road and sometimes reverse
            res2 = 200.0 - res2
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
            res1 = 200.0 - res1 #just a nuance due to having sometimes 0 as road and sometimes reverse
            res2 = 200.0 -res2
            res1 = numpy.multiply(res1, params[0])
            res2 = numpy.multiply(res2,params[1])
        else:
            res1 = res1.astype(float)
            res2 = res2.astype(float)
            res1 = 200.0 - res1
            res2 = 200.0 - res2
            res1 = numpy.multiply(res1,params[0])
            res2 = numpy.multiply(res2,params[1])

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

        rev_combp = 200 - combp
        if save_dir != 'unused':
            misc.toimage(rev_combp, cmin=0, cmax=255).save(save_dir+"sftmx_results/"+tile+".png")

        #print(numpy.max(combp))
        #Thresholding to get predictions
        combp[combp < params[-1]*2] = 1
        combp[combp >= params[-1]*2] = 0
        #print(numpy.max(combp))
        #print(numpy.min(combp))
        diff_sum = numpy.count_nonzero(combp!=target)
        combp[combp==1] = 2
        intersection = numpy.count_nonzero(combp==target)
        combp[combp==2] = 1
        if save_dir != 'unused':
            misc.toimage(combp, cmin=0, cmax=255).save(save_dir+"pred_results/"+tile+".png")

        true_neg = numpy.count_nonzero(combp==target) - intersection
        union = diff_sum + intersection
        total_intersection += intersection
        total_union += union
        total_neg += true_neg
        total_diff += diff_sum

    #print(total_intersection)
    #print(total_union)
    #print(total_diff)
    #print(total_neg)
    iou = total_intersection/total_union
    print(iou)
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
#    with localsolverblackbox.LocalSolverBlackBox() as ls:
#        model = ls.get_model()
#        p = model.float(40,60)
#        w1 = model.float(0.1,0.9)
#        w2 = model.float(0.1,0.9)
#        f = model.create_native_function(branin_eval)
#        call = model.call()
#        call.add_operand(f)
#        call.add_operand(p)
#        call.add_operand(w1)
#        call.add_operand(w2)
#        model.add_objective(call, localsolverblackbox.LSBBObjectiveDirection.MAXIMIZE)
#        model.close()
#        ls.get_param().set_evaluation_limit(50)
#        ls.solve()
#        sol = ls.get_solution()
#        print("p=" + str(sol.get_value(p)))
#        print("w1=" + str(sol.get_value(w1)))
#        print("w2=" + str(sol.get_value(w2)))
#        print("obj:" + str(sol.get_value(call)))
    # bounds for the grid search
    bound_w = [(0.0, 1.0)  for _ in range(n_members)]
    bound_w.append((40,60))
    # arguments to the loss function
    #search_arg = (n_members)
    # global optimization of ensemble weights
    start = time.time()
    result = differential_evolution(loss_function, bound_w, maxiter=10, tol=1e-5)
    # get the chosen weights
    end = time.time()
    params = result['x']
    weights = normalize(params[:-1])
    percent = float(params[-1])
    print('Optimized Weights: %s' % weights)
    # evaluate chosen weights
    #params = [0.5,0.5,45]
    params_final = list(weights)
    params_final.append(percent)
    score = compute_metric(params_final)
    print('Optimized Weights Score: %.3f' % score)
    score = compute_metric(params_final, "/home/ahana/road_data/test.txt")
    print('Optimized Weights Score: %.3f' % score)
    print("Time taken: "+str(end - start))


