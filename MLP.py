import math
import random
import data as d
from utils import *


#a single layer of a neural network
class Dense_Layer:
    def __init__ (self, n_inputs, n_neurons):
        #he initialized weights due to ReLU
        self.weights = [
            [random.gauss(0, math.sqrt(2.0 / n_inputs)) for i in range(n_neurons)] for i in range(n_inputs)]
        #set all biases in array to standard 0 for each neuron
        self.biases = [0 for i in range(n_neurons)]

    def forward (self, inputs):
        #save inputs for backward pass
        self.inputs = inputs
        #multiply the inputs by the weights and add the biases for each batch
        self.outputs = [add_bias(dot_product(batch, self.weights), self.biases) for batch in inputs]

    def backward(self, dvalues):
        #calc the  partial derivative of each value
        self.dweights = matrix_multiply(transpose(self.inputs), dvalues)
        self.dbiases = [sum(d) for d in transpose(dvalues)]
        self.dinputs = matrix_multiply(dvalues, transpose(self.weights))


#activation function for hidden layers to introduce nonlinearity and eliminate overflow chances
class ReLU:
    def forward(self, inputs):
        #Save inputs for backward pass
        self.inputs = inputs
        #Checks negative values in nested list and sets them to zero
        self.outputs = [[max(0, i) for i in batch] for batch in inputs]

    def backward(self, dvalues):
        #if the value is less then 0 set it to 0 else keep it the same
        self.dinputs = [[d if i > 0 else 0 for d, i in zip(l, p)] for l, p in zip(dvalues, self.inputs)]


#activation function for output layer
class softmax:
    def forward(self, inputs):
        # Normilize inputs to prevent overflowing by subtracting maxium in the batch
        norm_inputs = [[batch[i] - max(batch) for i in range(len(batch))] for batch in inputs]
        #divide eulur's number to the i by the sum of all i values in the batch
        self.outputs = [[math.exp(i) / sum([math.exp(i) for i in batch]) for i in batch] for batch in norm_inputs]

    def backward(self, y_true):
        #calculate the deritive of the inputs (and the loss of those inputs)
        self.dinputs = [[i - l for i, l in zip(z, y)] for z, y in zip(self.outputs, y_true)]


#Categorical Cross-entropy with One Hot encoded data
class CCE:
    def __call__(self, inputs, expected, epsilon = 1e-8):
        #clip values to prevent log of 0 errors in each batch
        clipped = [[max(epsilon, min(i, 1 - epsilon)) for i in batch] for batch in inputs]
        loss_matrix = [sum([-math.log(l) if e == 1 else 0 for l, e in zip(b, e)]) for b, e in zip(clipped, expected)]
        #return the mean of the array
        return sum(loss_matrix) / len(loss_matrix)


#optimizer
class Adam:
    def __init__(self, lr=0.01, b1=0.9, b2=0.999, decay = 0):
        self.a, self.ilr, self.decay = lr, lr, decay
        self.t = 0
        self.b1, self.b2 = b1, b2
        self.cached = {}



    def update(self, Layer, epilson = 1e-8):
        #save the matrix values for each layer in a cache
        if Layer not in self.cached:
            self.cached[Layer] = {
                "mw": [[0 for i in l]for l in Layer.weights],
                "vw": [[0 for i in l]for l in Layer.weights],
                "mb": [0] * len(Layer.biases),
                "vb": [0] * len(Layer.biases)
            }
        #increase the step count by 1
        self.t += 1
        #if decay is > 0 then add decay
        if self.decay:
            self.a = self.ilr * (1. / (1. + self.decay * self.t))

        #retrieve cached values
        mw = self.cached[Layer]["mw"]
        vw = self.cached[Layer]["vw"]
        mb = self.cached[Layer]["mb"]
        vb = self.cached[Layer]["vb"]

        #calc the new momentums and velocities
        mw = [[self.b1 * m + (1 - self.b1) * i for i, m in zip(l, i)] for l, i in zip(Layer.dweights, mw)]
        vw = [[self.b2 * v + (1 - self.b2) * (i ** 2) for i, v in zip(l, i)] for l, i in zip(Layer.dweights, vw)]

        mb = [self.b1 * m + (1 - self.b1) * i for i, m in zip(Layer.dbiases, mb)]
        vb = [self.b2 * v + (1 - self.b2) * (i ** 2) for i, v in zip(Layer.dbiases, vb)]

        #calc the corrected ms and vs
        mw_hat = [[m / (1 - self.b1 ** self.t) for m in batch] for batch in mw]
        vw_hat = [[v / (1 - self.b2 ** self.t) for v in batch] for batch in vw]

        mb_hat = [m / (1 - self.b1 ** self.t) for m in mb]
        vb_hat = [v / (1 - self.b2 ** self.t) for v in vb]

        #update weights and biases (epilson to stop a divide by 0 error)
        Layer.weights = [[w - self.a * m / (math.sqrt(v) + epilson) for w, m, v in zip(l, b, i)]
                            for l, b, i in zip(Layer.weights, mw_hat, vw_hat)]
        Layer.biases = [b - self.a * m / (math.sqrt(v) + epilson)
                            for b, m, v in zip(Layer.biases, mb_hat, vb_hat)]
        #save the new momentums and velocities
        self.cached[Layer] = {"mw": mw, "vw": vw, "mb": mb, "vb": vb}


#overall model object that holds all data of the MLP
class Model:
    def __init__ (self, steps, learning_rate, decay):
        #initialize each layer and activation function for forward and backward passes
        self.steps = steps
        #set the optimizer to Adam and pass the learning and decay rates
        self.optimizer = Adam(lr = learning_rate, decay = decay)
        #set the loss function to Categorical Cross Entropy or Log Loss
        self.loss_function = CCE()

    def forward(self, X_batch):
        #set the X_batch to inputs inorder to loop through each layer
        inputs = X_batch
        #call each step's forward method and set it's output to inputs
        for step in self.steps:
            step.forward(inputs)
            inputs = step.outputs
        #save the softmax activation function's output for loss calc
        self.SMoutputs = inputs

    def backward(self, y):
        #save outputs to y, Softmax's outputs
        outputs = y
        #reverse the list and iterate through each class, calling the backward method and saving the derivative
        for step in self.steps[::-1]:
            step.backward(outputs)
            outputs = step.dinputs

    def update(self):
        #iterate through the steps, of the class is a dense layer then update the weights * biases using Adam
        for i in self.steps:
            if isinstance(i, Dense_Layer):
                self.optimizer.update(i)

    def train(self, Dataset, epochs, batch_size):
        X, y = Dataset.train()

        for epoch in range(epochs):
            #calcs how many iterations to go through inorder to complete one epoch
            for i in range(math.floor(len(X) / batch_size)):
                #create a batch of data from the dataset and its corresponding truth values
                X_train, y_train = batch(X, batch_size, i), batch(y, batch_size, i)
                #runs through the neural network and updates the weights and biases
                self.forward(X_train)
                self.backward(y_train)
                self.update()
            #prints the loss of each specified epoch
            print(f'Epoch {epoch + 1}, Loss: {self.loss_function(self.SMoutputs, y)} lr: {self.optimizer.a} '
                  f'steps {self.optimizer.t}')

    def evaluate(self, Dataset):
        #get the testing dataset and set it's truth values
        X, y_true = Dataset.test()
        #forward propagation inorder to determine what the ANN thinks
        self.forward(X)
        #takes an argmax of the softmax outputs and the y_true matrixs. If they are equal, add one and sum the list
        correct = sum([1 if s.index(max(s)) == y.index(max(y)) else 0 for s, y in zip(self.SMoutputs, y_true)])
        #divide the number correct by the number of data
        print(correct / len(y_true))


