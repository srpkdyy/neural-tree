import random
import numpy as np
from operator import itemgetter



class Neuron(object):
    def __init__(self, idx, W, y):
        self.idx = idx
        self.W = W
        self.label = y
        self.children = list()
        self.n_updated = 1


    def distance(self, e):
            return np.sqrt(np.sum(np.square(self.W - e)))


    def connected(self, n):
        self.children.append(n)


    def update(self, e):
        self.W = self.W + (e - self.W)/(self.n_updated + 1)
        self.n_updated += 1



class NeuralTree(object):
    def __init__(self, n_estimators=1, threshold=0, random_state=None, verbose=0):
        self.n_estimators = n_estimators
        self.threshold = threshold
        self.verbose = verbose
        self.estimators = list()

        if not random_state is None:
            random.seed(random_state)
            np.random.seed(random_state)


    def train(self, E):
        N = E.shape[0]
        root = Neuron(0, *E[0])
        j = 1
        for i in range(1, N):
            e = E[i]

            minimumDistance = root.distance(e[0])
            updated_n = list()
            winner = root
            minimumDistance, updated_n, winner = self.test(e[0], root, minimumDistance, updated_n)

            if minimumDistance >= self.threshold:
                if not winner.children:
                    n = Neuron(j, winner.W, winner.label)
                    winner.connected(n)
                    j += 1
            
                n = Neuron(j, *e)
                winner.connected(n)
                j += 1
        
            for n in updated_n:
                n.update(e[0])

        return root


    def test(self, e, subRoot, minDist, updated_n):
        updated_n.append(subRoot)
        winner = subRoot

        for n in subRoot.children:
            dist = n.distance(e)

            if dist < minDist:
                minDist, updated_n, winner = self.test(e, n, dist, updated_n)

        return minDist, updated_n, winner


    def fit(self, X, y, shuffle=True, bagging=False):
        ds = list(zip(X, y))
        for i in range(self.n_estimators):
            if shuffle: 
                np.random.shuffle(ds)

            _X = ds
            if bagging:
                _X = random.choices(ds, k=len(ds))

            tree = self.train(np.array(_X))
            self.estimators.append(tree)

            if self.verbose != 0:
                print('====== Tree No.{} ======='.format(i+1))
                self.p_tree(tree)
        return self


    def predict(self, X):
        X = np.array(X)
        assert X.ndim >= 2, 'x dim must be >= 2'

        pre_y = list()
        for e in X:
            pre_labels = np.array([], dtype=int)
            for subRoot in self.estimators:
                while subRoot.children:
                    dist = [[n.distance(e), n] for n in subRoot.children]
                    dist.sort(key=itemgetter(0))
                    subRoot = dist[0][1]
                pre_labels = np.append(pre_labels, subRoot.label)
            # 最頻値
            count = np.bincount(pre_labels)
            pre_y.append(np.argmax(count))
        return pre_y


    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


    def p_tree(self, n):
        print('idx ==>'+str(n.idx))
        print('W ==>'+str(n.W))
        print('i ==>'+str(n.n_updated))
        print('label ==>'+str(n.label))
        print('children ==>', end='')
        for c in n.children:
            print(str(c.idx), end=' ')
        print(end='\n\n')
    
        for c in n.children:
            self.p_tree(c)



if __name__ == '__main__':
    E = [[1], [2], [3], [4]]
    nt = NeuralTree(n_estimators=1, verbose=1)
    nt.fit(E, E, shuffle=False)
    print(nt.predict([[3.1]]))

