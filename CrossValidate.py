import numpy as np

def shuffle_in_unison(data, labels):
    m = data.shape[0]
    r = data.shape[1]
    c = labels.shape[1]
    new_data = np.zeros((m,r))
    new_labels = np.zeros((m,c))
    perm = np.random.permutation(m)
    for i in range(m):
        new_data[i,:] = data[perm[i],:]
        new_labels[i,:] = labels[perm[i],:]
    return new_data, new_labels

def split_training_data(data,labels,k=5):
    m = data.shape[0]
    r = data.shape[1]

    X,y = shuffle_in_unison(data,labels)

    size = int(m / float(k))
    for i in range(k):
        X_train = X[:i * size, :]
        y_train = y[:i * size, :]
        X_test = X[i * size : (i + 1) * size, :]
        y_test = y[i * size : (i + 1) * size, :]
        X_train = np.concatenate((X_train, X[(i + 1) * size:, :]), axis=0)
        y_train = np.concatenate((y_train, y[(i + 1) * size:, :]), axis=0)
        yield X_train, y_train, X_test, y_test

if __name__ == '__main__':
    m = 100
    r = 20
    c = 2
    data = np.random.randn(m,r)
    labels = np.random.randn(m,2)
    for X_train,Y_train,X_test,Y_test in split_training_data(data,labels):
        print(X_train.shape, Y_train.shape)
        print(X_test.shape, Y_test.shape)
