from grid_search import compute_metric
import sys

if __name__ == "__main__":
    with open('weights1.txt') as f:
        params = f.read().splitlines()

    params_final = [float(x) for x in params]

    score = compute_metric(params_final, "/home/ahana/road_data/test.txt")
    print('Optimized Weights Score: %.3f' % score)
