from params import *
from model import *

# innizlize each layer and activation function
STEPS = [
   DenseLayer(784, 100), ReLU(), DropoutLayer(dr), DenseLayer(100, 64), ReLU(), DropoutLayer(dr),
   DenseLayer(64, 10), Softmax()
]

# create the model object
model = Model(STEPS, scheduler = OneCycleLR(max_lr, lr, EPOCHS, BATCH_SIZE, Dataset.size))

# train the model
model.train(Dataset, EPOCHS, BATCH_SIZE)

# test the model to determine accuracy
model.test(Dataset)

# save the model to the specified file1
save(model, 'model.pickle')
