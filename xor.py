from neural_tree import NeuralTree


def main():
    X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [0, 1, 1, 0]
    
    X_test = [[0.1, 0.1], [0.9, 0.1], [0.45, 0.55]]

    nt = NeuralTree(n_estimators=3, verbose=1)
    nt.fit(X_train, y_train)

    print(nt.predict(X_test))

    

if __name__ == '__main__':
    main()

