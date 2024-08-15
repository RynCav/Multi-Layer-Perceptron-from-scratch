#standerd MNIST dataset
class MNIST:
    def __init__(self):
        from utils import load
        directory = 'Datasets/MNIST/'
        #load files, values are already normilized & y_true is onehot encoded
        self.train_X, self.test_X = load(f'{directory}MNIST_Train_X'), load(f'{directory}MNIST_Test_X')
        self.train_y, self.test_y = load(f'{directory}MNIST_Train_y'), load(f'{directory}MNIST_Test_y')

    def train(self):
        #returns the training values
        return self.train_X, self.train_y

    def test(self):
        #returns the testing data
        return self.test_X, self.test_y


#Fashion MNIST dataset
class FashionMNIST:
    def __init__(self):
        from utils import load
        directory = 'Datasets/Fashion MNIST/'
        #load files, values are already normilized & y_true is onehot encoded
        self.train_X, self.test_X = load(f'{directory}FMNIST_X_test'), load(f'{directory}FMNIST_X_train')
        self.train_y, self.test_y = load(f'{directory}FMNIST_y_test'), load(f'{directory}FMNIST_y_train')

    def train(self):
        #returns the training data
        return self.train_X, self.train_y

    def test(self):
        #returns the testing data
        return self.test_X, self.test_y









