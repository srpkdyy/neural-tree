import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from neural_tree import NeuralTree



def main():
    seed = 1711
    print('seed: {}\n'.format(seed))

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=seed)

    for n in [1, 3, 5, 7, 9, 11, 15, 25]:
        print('===== Estimators : {} ====='.format(n))

        nt = NeuralTree(n_estimators=n, random_state=seed)
        nt.fit(X_train, y_train)

        pred = nt.predict(X_test)

        correct = np.count_nonzero(pred == y_test)
        acc = 100. * correct / len(y_test)

        print('Acc: {}% ({}/{})\n'.format(acc, correct, len(y_test)))



if __name__ == '__main__':
    main()

