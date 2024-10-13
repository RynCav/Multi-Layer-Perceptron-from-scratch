# Multi-Layer-Perceptron From Scratch 

A Multilayer Perceptron implementation in Python without external Libraries. 

The following model got 97% accuracy on Fashion MNIST:

- Dense Layer: 784 input neurons, 100 hidden neurons

- ReLU

- Dropout Layer 0.3

- Dense Layer: 100 input neurons, 64 hidden neurons

- ReLU

- Dropout Layer 0.3

- Dense Layer: 64 input neurons, 10 output neurons
  
- Softmax

# Built in Classes

**Implemented Activation Functions:**
- Sigmoid 
- ReLU (Rectified Linear Unit)
- Tanh
- Softmax

**Implemented Loss Functions:**
- Categorical Cross-Entropy
-  MSE (Mean Squared Error)

**Implemented Regulation:**
- Dropout Layers
- L1 & L2 Regulations

**Implemented Optimizers:**
- Adam (Adaptive Moment Estimation)

**Learning Rate Schedulers:**
- OneCycleLR
- InverseTimeDecay

**Built in Datasets:**
- MNIST
- Fashion MNIST

EarlyStopping available for Datasets with validation sets
