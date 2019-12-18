from grid_search_avg import compute_metric

if __name__ == "__main__":
    with open('weights.txt') as f:
        params = f.read().splitlines()

     score = compute_metric(params_final, "/home/ahana/road_data/large_dataset/val.txt", sys.srgv[1])
     print('Optimized Weights Score: %.3f' % score)
