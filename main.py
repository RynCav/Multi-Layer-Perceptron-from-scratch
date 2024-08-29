from params import *
from model import *

#innizlize each layer and activation function
STEPS = [
   Dense_Layer(784, 100), ReLU(), Dropout_Layer(dr), Dense_Layer(100, 64), ReLU(), Dropout_Layer(dr),
   Dense_Layer(64, 10), Softmax()
]

#create the model object
model = Model(STEPS, lr, DECAY)

#train the model
model.train(dataset, EPOCHS, BATCH_SIZE)

#test the model to determine accuracy
model.evaluate(dataset)

#save the model to the specified file
save(model, 'model.pickle')
