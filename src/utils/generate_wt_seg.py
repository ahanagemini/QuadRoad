from grid_search import compute_metric
import sys
"""
Code to compute IoU using computed/ saved weights and 
optionally save the generated segmentation
Args:
    paths_file: file with all the relevant paths as described 
        for compute_metric function 
    param_file: stores the parameters
"""
if __name__ == "__main__":
    paths_file = sys.argv[1]
    param_file = sys.argv[2]
    with open(param_file) as f:
        params = f.read().splitlines()

    params_final = [float(x) for x in params]

    score = compute_metric(params_final, paths_file)
    print('Optimized Weights Score: %.3f' % score)
