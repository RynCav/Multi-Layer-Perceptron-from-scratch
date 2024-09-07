from params import *
from model import *

#innizlize each layer and activation function
STEPS = [
   Dense_Layer(784, 100), ReLU(), Dropout_Layer(dr), Dense_Layer(100, 64), ReLU(), Dropout_Layer(dr),
   Dense_Layer(64, 10), Softmax()
]

#create the model object
model = Model(STEPS)

#train the model
model.train(dataset, EPOCHS, BATCH_SIZE)

#test the model to determine accuracy
model.test(dataset)

#save the model to the specified file1
save(model, 'model.pickle')
