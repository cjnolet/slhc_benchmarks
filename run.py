from slhc_benchmarks.datasets import get_dataset, DATASETS
from slhc_benchmarks.distance import dataset_transform
from cuml.cluster import AgglomerativeClustering as cuAgg
from sklearn.cluster import AgglomerativeClustering as skAgg

import sys, getopt, time


def time_train(dataset_name, model, X, env):
    s = time.time()
    model.fit(X)
    print("%s took %s seconds on %s" % (env, (time.time() - s), dataset_name))


def main(argv):
    dataset_name = ''
    env = ''
    opts, args = getopt.getopt(argv, "hd:e:", ["dataset=", "env="])
    for opt, arg in opts:
        if opt == "-h":
            print("run.py -d <dataset> -e <device|host|both>\n")
            print("Available datasets: %s" % list(DATASETS.keys()))
            sys.exit()
        elif opt in ("-d", "--dataset"):
            dataset_name = arg
        elif opt in ("-e", "--env"):
            env = arg

    print("Dataset: %s" % dataset_name)
    print("Env: %s" % env)

    D, dim = get_dataset(dataset_name)
    X_train, X_test = dataset_transform(D)

    if env == "device" or env == "both":
        time_train(dataset_name, cuAgg(linkage="single"), X_train, "device")

    if env == "host" or env == "both":
        time_train(dataset_name, skAgg(linkage="single"), X_train, "host")

if __name__ == "__main__":
    main(sys.argv[1:])