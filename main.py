import numpy as np



class Neuron(object):
    def __init__(self, idx, W):
        self.idx = idx
        self.W = W
        self.children = list()
        self.n_updated = 1

    def distance(self, e):
            return np.sqrt(np.sum(np.square(self.W - e)))


    def connected(self, n):
        self.children.append(n)


    def update(self, i, e):
        self.W = self.W + (e - self.W)/(self.n_updated + 1)
        self.n_updated += 1



def train(E, threshold=0):
    N = E.shape[0]
    threshold = 0

    root = Neuron(0, E[0])

    j = 1
    for i in range(1, N):
        e = E[i]

        minimumDistance = root.distance(e)
        updated_n = list()
        winner = root
        minimumDistance, updated_n, winner = test(e, root, minimumDistance, updated_n)

        if minimumDistance > threshold:
            if not winner.children:
                n = Neuron(j, winner.W)
                winner.connected(n)
                j += 1
            
            n = Neuron(j, e)
            winner.connected(n)
            j += 1
        
        for n in updated_n:
            n.update(i, e)
        
    return root



def test(e, subRoot, minDist, updated_n):
    updated_n.append(subRoot)
    winner = subRoot

    for n in subRoot.children:
        dist = n.distance(e)

        if dist < minDist:
            minDist, updated_n, winner = test(e, n, dist, updated_n)

    return minDist, updated_n, winner



def main():
    E = np.array([[1, 2, 3, 4], [4, 1, 2, 3], [1, 3, 2, 4]])
    
    trees = list()
    for e in E:
        print('Test case:'+str(e))
        tree = train(e, threshold=0)
        print_tree(tree)
        trees.append(tree)

    w_sum = 0
    for i, tree in enumerate(trees):
        *_, winner = test(1.6, tree, np.Infinity, [])
        print('winner' + str(i+1) + '.W ==>' + str(winner.W))
        w_sum += winner.W

    print('Ave ==>'+str(w_sum / (i+1)))



def print_neuron(n):
    print('idx ==>'+str(n.idx))
    print('W ==>'+str(n.W))
    print('i ==>'+str(n.n_updated))
    print('children ==>', end='')
    for c in n.children:
        print(str(c.idx), end=' ')
    print(end='\n\n')



def print_tree(n):
    print_neuron(n)
    
    for c in n.children:
        print_tree(c)



if __name__ == '__main__':
    main()
