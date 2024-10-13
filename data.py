from utils import load, randomize_lists


# inherited class, template
class Data:
    #initialize the data
    def __init__(self):
        #load files, values are already normilized & y_true is onehot encoded
        self.train_X, self.test_X = load(f'{self.directory}Train_X'), load(f'{self.directory}Test_X')
        self.train_y, self.test_y = load(f'{self.directory}Train_y'), load(f'{self.directory}Test_y')
        if self.valuation_set:
            self.evaluate_y, self.evaluate_y = load(f'{self.directory}Evaluate_y'), load(f'{self.directory}Evaluate_y')

    #load the training data
    def train(self):
        # Randomises the data
        return randomize_lists(self.train_X, self.train_y)

    def evaluate(self):
        return self.evaluate_X, self.evaluate_y

    #load the testing data
    def test(self):
        # returns the testing data
        return self.test_X, self.test_y


# standerd MNIST dataset
class MNIST(Data):
    directory = 'Datasets/MNIST/MNIST_'
    size = 60000
    valuation_set = False


# Fashion MNIST dataset
class FashionMNIST(Data):
    directory = 'Datasets/Fashion MNIST/FMNIST_'
    size = 60000
    valuation_set = False
