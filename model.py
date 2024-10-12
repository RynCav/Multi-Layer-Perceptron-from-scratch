import math
from utils import *

# a single layer of a neural network
class DenseLayer:
    def __init__(self, n_inputs, n_neurons, l1=0.00001, l2=0.00001):
        # he initialized weights due to ReLU
        self.weights = [
            [random.gauss(0, math.sqrt(2.0 / n_inputs)) for i in range(n_neurons)] for i in range(n_inputs)]

        # set all biases in array to standard 0 for each neuron
        self.biases = [0 for i in range(n_neurons)]

        # set reg factors
        self.l1_factor, self.l2_factor = l1, l2

    def forward(self, inputs):
        # save inputs for backward pass
        self.inputs = inputs
        # multiply the inputs by the weights and add the biases for each batch
        outputs = [add_bias(dot_product(batch, self.weights), self.biases) for batch in inputs]
        return outputs

    def backward(self, dvalues):
        # calc the  partial derivative of each value
        self.dweights = matrix_multiply(transpose(self.inputs), dvalues)
        self.dbiases = [sum(d) for d in transpose(dvalues)]
        self.dinputs = matrix_multiply(dvalues, transpose(self.weights))

        # add the l1 gradients
        if self.l1_factor > 0:
            self.dweights = [[dw + self.l1_factor * (1 if w > 0 else -1) for dw, w in zip(drow, row)]
                             for drow, row in zip(self.dweights, self.weights)]
            self.dbiases = [db + self.l1_factor * (1 if b > 0 else -1) for db, b in zip(self.dbiases, self.biases)]

        # add the l2 gradients
        if self.l2_factor > 0:
            self.dweights = [[dw + 2 * self.l2_factor * w for dw, w in zip(drow, row)]
                             for drow, row in zip(self.dweights, self.weights)]
            self.dbiases = [db + 2 * self.l2_factor * b for db, b in zip(self.dbiases, self.biases)]

        return self.dinputs


# sets any negative value to 0
class ReLU:
    def forward(self, inputs):
        # Save inputs for backward pass
        self.inputs = inputs
        # Checks negative values in nested list and sets them to zero
        outputs = [[max(0, i) for i in batch] for batch in inputs]
        return outputs

    def backward(self, dvalues):
        # if the value is less then 0 set it to 0 else keep it the same
        dinputs = [[d if i > 0 else 0 for d, i in zip(l, p)] for l, p in zip(dvalues, self.inputs)]
        return dinputs


# scales the values between 1 & 0
class Sigmoid:
    def forward(self, inputs):
        self.outputs = [[1 / (1 + math.exp(-value)) for value in batch] for batch in inputs]
        return self.outputs

    def backward(self, dvalues):
        dinputs = [[output * (1 - output) * dvalue for output, dvalue in zip(batch, dbatch)]
                        for batch, dbatch in zip(self.outputs, dvalues)]
        return dinputs


# SCales the values between -1 & 1
class Tanh:
    def forward(self, inputs):
        self.inputs = inputs
        outputs = [[math.tanh(i) for i in batch] for batch in inputs]
        return outputs

    def backward(self, dvalues):
        dinputs = [[d * (1 - math.tanh(i) ** 2) for i, d in zip(batch, dbatch)] for batch, dbatch in
                   zip(self.inputs, dvalues)]
        return dinputs


# Dropout Layer
class DropoutLayer:
    def __init__(self, rate=0.5):
        self.rate = rate

    def forward(self, inputs):
        self.inputs = inputs
        self.bmask = [[1 if random.random() < self.rate else 0 for i in row] for row in inputs]
        outputs = [[i * m for i, m in zip(irow, mrow)] for irow, mrow in zip(inputs, self.bmask)]
        return outputs

    def backward(self, dvalues):
        dinputs = [[d * m for d, m in zip(drow, mrow)] for drow, mrow in zip(dvalues, self.bmask)]
        return dinputs


# activation function for output layer
class Softmax:
    def forward(self, inputs):
        # Normilize inputs to prevent overflowing by subtracting maxium in the batch
        norm_inputs = [[batch[i] - max(batch) for i in range(len(batch))] for batch in inputs]
        # divide eulur's number to the i by the sum of all i values in the batch
        self.outputs = [[math.exp(i) / sum([math.exp(i) for i in batch]) for i in batch] for batch in norm_inputs]
        return self.outputs

    def backward(self, y_true):
         # calculate the deritive of the inputs (and the loss of those inputs)
         return [[i - l for i, l in zip(z, y)] for z, y in zip(self.outputs, y_true)]


# Categorical Cross-entropy with One Hot encoded data
class _RegLoss:
    def regulazation_loss(self, model):
        reg_loss = 0
        for step in model.steps:
            if isinstance(step, DenseLayer):
                if step.l1_factor > 0:
                    reg_loss += step.l1_factor * sum(sum(abs(w) for w in row) for row in step.weights)
                if step.l2_factor > 0:
                    reg_loss += step.l2_factor * sum(sum(w ** 2 for w in row) for row in step.weights)
        return reg_loss


class CCE(_RegLoss):
     def forward(self, model, predicated, expected, epsilon=1e-8):
         # clip values to prevent log of 0 errors in each batch
         clipped = [[max(epsilon, min(i, 1 - epsilon)) for i in batch] for batch in predicated]
         loss_matrix = [sum([-math.log(l) if e == 1 else 0 for l, e in zip(b, e)]) for b, e in
                       zip(clipped, expected)]
         # calc & return the losses and add reg loss
         norm_loss, reg_loss = sum(loss_matrix) / len(loss_matrix), self.regulazation_loss(model)
         return f'Loss: {norm_loss + reg_loss} (Normal Loss: {norm_loss} Reg Loss: {reg_loss})'


     def backward(self, predicated, expected, epsilon=1e-8):
         clipped = [[max(epsilon, min(i, 1 - epsilon)) for i in batch] for batch in predicated]
         dinputs = [[(c - e) for c, e in zip(cl, ex)] for cl, ex in zip(clipped, expected)]
         return dinputs


class MSE(_RegLoss):
    def forward(self, model, predicted, expected):
        norm_loss = sum([sum([(y - y_hat) ** 2 for y, y_hat in zip(batch, batch_hat)]) / len(batch)
                         for batch, batch_hat in zip(predicted, expected)]) / len(expected)
        reg_loss = self.regulazation_loss(model)
        return f'Loss: {norm_loss + reg_loss} (Normal Loss: {norm_loss} Reg Loss: {reg_loss})'

    def backward(self, predicted, expected):
        N = len(expected)
        dinputs = [[(2 / N) * (y_hat - y) for y_hat, y in zip(batch_hat, batch)]
                        for batch_hat, batch in zip(predicted, expected)]
        return dinputs


# optimizer
class Adam:
    def __init__(self, b1=0.9, b2=0.999):
        self.t = 0
        self.b1, self.b2 = b1, b2
        self.cached = {}

    def update(self, Layer, lr, epsilon=1e-8):
        # Initialize cache if not already done
        if id(Layer) not in self.cached:
            self.cached[id(Layer)] = {
                "mw": [[0 for i in l] for l in Layer.weights],
                "vw": [[0 for i in l] for l in Layer.weights],
                "mb": [0] * len(Layer.biases),
                "vb": [0] * len(Layer.biases)
           }

        # Increment step count
        self.t += 1

        # Retrieve cached momentums and velocities
        mw = self.cached[id(Layer)]["mw"]
        vw = self.cached[id(Layer)]["vw"]
        mb = self.cached[id(Layer)]["mb"]
        vb = self.cached[id(Layer)]["vb"]


        # Update momentums and velocities
        mw = [[self.b1 * m + (1 - self.b1) * dw for m, dw in zip(mw, dweights)]
              for mw, dweights in zip(mw, Layer.dweights)]
        vw = [[self.b2 * v + (1 - self.b2) * (dw ** 2) for dw, v in zip(dweights, vw)]
              for dweights, vw in zip(Layer.dweights, vw)]


        mb = [self.b1 * mb[i] + (1 - self.b1) * Layer.dbiases[i] for i in range(len(mb))]
        vb = [self.b2 * vb[i] + (1 - self.b2) * (Layer.dbiases[i] ** 2) for i in range(len(vb))]


        # Corrected momentums and velocities
        mw_hat = [[m / (1 - self.b1 ** self.t) for m in mw_row] for mw_row in mw]
        vw_hat = [[v / (1 - self.b2 ** self.t) for v in vw_row] for vw_row in vw]
        mb_hat = [m / (1 - self.b1 ** self.t) for m in mb]
        vb_hat = [v / (1 - self.b2 ** self.t) for v in vb]


        # Update weights and biases
        Layer.weights = [[w - lr * m / (math.sqrt(v) + epsilon) for w, m, v in zip(weights, mh, vh)]
                         for weights, mh, vh in zip(Layer.weights, mw_hat, vw_hat)]


        Layer.biases = [b - lr * m / (math.sqrt(v) + epsilon) for b, m, v in
                        zip(Layer.biases, mb_hat, vb_hat)]


        # Save the updated momentums and velocities in the cache
        self.cached[id(Layer)] = {"mw": mw, "vw": vw, "mb": mb, "vb": vb}


# stops the model early when necessary
class EarlyStopping:
    def __init__(self, patience):
        self.patience, self.patience_counter = patience, 0
        self.best_model = None
        self.best_loss = float('inf')

    def __call__(self, model, dataset):
        new_loss = model.evaluate(dataset)
        if new_loss < self.best_loss:
            self.best_loss = new_loss
            self.best_model = model
        else:
            self.patience_counter += 1
            if self.patience_counter == self.patience:
                self.stop(model)

    def stop(self, model):
       raise Exception("early stopping activated")
       model.stop = True


#increase then slowley decrease the lr fro stability
class OneCycleLR:
    def __init__(self, max_lr, initial_lr, epochs, batch_size, dataset_size):
        self.total_steps = dataset_size / batch_size * epochs
        self.warmup_steps = self.total_steps * 0.1
        self.anneal_steps = self.total_steps - self.warmup_steps
        self.max_lr, self.initial_lr = max_lr, initial_lr
        self.lr = self.initial_lr

    def step(self, step):
        if step <= self.warmup_steps:
            self.warm_up(step)
        else:
            self.aneeling(step)
        return self.lr

    def warm_up(self, step):
        self.lr = self.initial_lr + (step / self.warmup_steps) * (self.max_lr - self.initial_lr)

    def aneeling(self, step):
        anneal = (step - self.warmup_steps) / self.anneal_steps
        self.lr = self.max_lr - (self.max_lr - self.initial_lr) * anneal


#Slowely decay the lr lineary
class InverseTimeDecay:
    def __init__(self, lr, decay=0):
        self.lr, self.initial_lr = lr, lr
        self.decay = decay

    def step(self, steps):
        if self.decay:
            self.lr = self.initial_lr / (1 / (1 + self.decay * steps))
        return self.lr


# overall model object that holds all data of the MLP
class Model:
    def __init__(self, layers=[], optimizer=Adam(), scheduler=InverseTimeDecay(0.01), early_stopping=None):
        # initialize each layer and activation function for forward and backward passes
        self.steps = layers
        # set the optimizer to Adam and pass the learning and decay rates
        self.optimizer = optimizer
        # set the loss function to Categorical Cross Entropy or Log Loss
        self.loss_function = CCE()
        # set whether early stopping is applicable
        self.early_stopping = early_stopping
        self.stop = False
        # set the scheduler
        self.scheduler = scheduler

    def add(self, new_layer):
       self.Layers.append(new_layer)

    def update(self):
        # iterate through the steps, of the class is a dense layer then update the weights * biases using Adam
        for i in self.steps:
            if isinstance(i, DenseLayer):
                self.optimizer.update(i, self.scheduler.step(self.optimizer.t))

    def train(self, dataset, epochs, batch_size):
        x, y = dataset.train()
        for epoch in range(epochs):
            # Calculate how many iterations to go through in one epoch
            for i in range(math.ceil(len(x) / batch_size)):
                # Create a batch of data and its corresponding truth values
                x_train, y_train = batch(x, batch_size, i), batch(y, batch_size, i)

                # Forward pass: Run through the network
                outputs = self.n_pass(self.steps, 'forward', x_train, True)

                # Calculate loss gradients
                dinputs = self.loss_function.backward(outputs, y_train)

                # Backward pass of the network
                self.n_pass(self.steps[::-1], 'backward', dinputs, True)

                # Update weights and biases using the optimizer
                self.update()

            #print infomation
            print(f'Epoch: {epoch + 1} {self.loss_function.forward(self, outputs, y_train)} Accuracy: '
                    f'{self.accuracy(dinputs, y_train)}')

            if self.early_stopping:
                self.early_stopping(self, dataset)
                if self.stop:
                    break

    def validate(self, dataset):
        # get the validation dataset and set it's truth values
        x, y_true = dataset.validate()
        # forward propagation inorder to determine what the ANN thinks
        self.n_pass(self.steps, 'forward', x, False)
        return self.loss_function.forward(self, y_true)

    def test(self, dataset):
        # get the testing dataset and set it's truth values
        x, y_true = dataset.test()
        # forward propagation inorder to determine what the ANN thinks
        inputs = self.n_pass(self.steps, 'forward', x, False)
        # print out results
        print(f'Testing: {self.loss_function.forward(self, y_true)} lr: {self.scheduler.lr} '
                f'steps {self.optimizer.t} Accuracy: {self.accuracy(inputs, y_true)}')

    @staticmethod
    def accuracy(y_pred, y_true):
        # takes an argmax of the softmax outputs and the y_true matrixs. If they are equal, add one and sum the list
        correct = sum([1 if argmax(s) == argmax(y) else 0 for s, y in zip(y_pred, y_true)])
        # divide the number correct by the number of data
        return correct / len(y_true)

    @staticmethod
    def n_pass(steps, pass_type, x_batch, training):
        # set the X_batch to inputs inorder to loop through each layer
        inputs = x_batch
        # call each step's forward method and set it's output to inputs
        for step in steps:
            # turns off Dropout Layers when on testing set or validation set
            if training or not isinstance(step, DropoutLayer):
                inputs = getattr(step, pass_type)(inputs)
        return inputs
