import copy
import argparse
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold

from neural_tree import NeuralTree



def main(args):
    print('Random seed: %d\n' % args.seed)
    n_splits = 10

    iris = load_iris()
    skf = StratifiedKFold(n_splits=n_splits, random_state=args.seed, shuffle=True)

    for n in [1, 3, 5, 7, 9, 11, 15, 25]:
        scores = np.empty(n_splits)

        for i, (train_idx, test_idx) in enumerate(skf.split(iris.data, iris.target)):
            X_train, X_test = iris.data[train_idx], iris.data[test_idx]
            y_train, y_test = iris.target[train_idx], iris.target[test_idx]

            nt = NeuralTree(n_estimators=n, random_state=args.seed).fit(X_train, y_train)
            scores[i] = nt.score(X_test, y_test)

        print('===== Estimators : %2d =====' % n)
        print('Accuracy: %0.3f (+/- %0.3f)\n' % (scores.mean(), scores.std() * 2))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', metavar='N', type=int, default=1,
            help='Random seed value (default: 1)')
    #parser.add_argument('--n-splits', metavar='N', type=int, default=10,
    #        help='Number to be spilit')

    main(parser.parse_args())

