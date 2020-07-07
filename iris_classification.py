import argparse
import copy
import json
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

    n_estimators = [1, 3, 5, 7, 9, 11, 15, 25]
    result = {
            'n_estimators': n_estimators,
            'scores': list(),
            'pr_scores': list(),
            'n_units': list(),
            'pr_n_units': list()
    }
    for n in n_estimators:
        scores = np.empty(n_splits)
        n_neurons = np.empty(n_splits)
        pruned_scores = np.empty(n_splits)
        pruned_n_neurons = np.empty(n_splits)

        for i, (train_idx, test_idx) in tqdm(enumerate(skf.split(iris.data, iris.target))):
            X_train, X_test = iris.data[train_idx], iris.data[test_idx]
            y_train, y_test = iris.target[train_idx], iris.target[test_idx]

            nt = NeuralTree(n_estimators=n, random_state=args.seed)
            nt.fit(X_train, y_train, bagging=True)

            scores[i] = nt.score(X_test, y_test)
            n_neurons[i] = np.sum(nt.n_neurons)

            nt.prune(alpha=1.0)

            pruned_scores[i] = nt.score(X_test, y_test)
            pruned_n_neurons[i] = np.sum(nt.n_neurons)

        print('\033[32m===== Estimators : %2d =====\033[0m' % n)
        print('Accuracy: %0.3f (+/- %0.3f)' % (scores.mean(), scores.std() * 2))
        print('Accuracy(pruned): %0.3f (+/- %0.3f)' % (pruned_scores.mean(), pruned_scores.std() * 2))
        print('n_units: %0.1f (+/- %0.1f)' % (n_neurons.mean(), n_neurons.std() * 2))
        print('n_units(pruned): %0.1f (+/- %0.1f)\n' % (pruned_n_neurons.mean(), pruned_n_neurons.std() * 2))

        result['scores'].append(scores.mean())
        result['pr_scores'].append(pruned_scores.mean())
        result['n_units'].append(n_neurons.mean())
        result['pr_n_units'].append(pruned_n_neurons.mean())

    if args.out:
        with open('result.json', 'w') as f:
            json.dump(result, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', metavar='N', type=int, default=None,
            help='Random seed value (default: None)')
    parser.add_argument('--out', action='store_true',
            help='Whether to output the results as a json file')
    #parser.add_argument('--n-splits', metavar='N', type=int, default=10,
    #        help='Number to be spilit')

    main(parser.parse_args())

