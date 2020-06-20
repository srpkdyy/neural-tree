import copy
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold

from neural_tree import NeuralTree



def main(args):
    print('Random seed: {}\n'.format(args.seed))
    n_splits = 10

    iris = load_iris()
    skf = StratifiedKFold(n_splits=n_splits, random_state=args.seed, shuffle=True)

    for n in [1, 3, 5, 7, 9, 11, 15, 25]:
        scores = np.empty(n_splits)

        for i, (train_idx, test_idx) in tqdm(enumerate(skf.split(iris.data, iris.target))):
            X_train, X_test = iris.data[train_idx], iris.data[test_idx]
            y_train, y_test = iris.target[train_idx], iris.target[test_idx]

            nt = NeuralTree(n_estimators=n, random_state=args.seed)
            nt.fit(X_train, y_train, bagging=True)

            scores[i] = nt.score(X_test, y_test)

        print('\033[32m===== Estimators : %2d =====\033[0m' % n)
        print('Accuracy: %0.3f (+/- %0.3f)\n' % (scores.mean(), scores.std() * 2))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', metavar='N', type=int, default=None,
            help='Random seed value (default: None)')
    #parser.add_argument('--n-splits', metavar='N', type=int, default=10,
    #        help='Number to be spilit')

    main(parser.parse_args())

