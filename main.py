from MLP import *

#Hyperparameters
BATCH_SIZE = 10
EPOCHS = 1
lr = 0.001
DECAY = 0.0001

#Dataset used:
dataset = d.FashionMNIST()

model = Model([Dense_Layer(784, 100), ReLU(), Dense_Layer(100, 64), ReLU(), Dense_Layer(64, 10), softmax()], lr, DECAY)
model.train(dataset, EPOCHS, BATCH_SIZE)
model.evaluate(dataset)
"""
# Load and evaluate model
model = load('model.pickle')
model.evaluate(dataset)
"""

