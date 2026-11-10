# Neural networks · Models

<!-- https://learn-inference.com/chapters/models/neural-networks -->

Strip away the vocabulary and a neural network is a stack of layers that each take a vector, multiply it by a matrix of learned weights, and pass the result on. An input layer, many hidden layers, an output layer.

### 2.1.1 Linear layers and matmul

A linear layer computes `y = Wx + b`: a matrix of weights times an input vector, plus a bias. That single operation is where essentially all of the arithmetic in inference happens, which is why GPUs (machines built to multiply matrices) turned out to be the right hardware for a problem nobody designed them for.

It also explains why parameter count is such a good proxy for cost. More parameters means bigger matrices, which means more multiply-accumulates and, more importantly, more bytes to move.

### 2.1.2 Activation functions

Stack two linear layers and you have accomplished nothing. The composition of two matrix multiplications is another matrix multiplication, so a hundred stacked linear layers collapse algebraically into one. Depth would buy no expressiveness at all.

An activation function (ReLU and its relatives) applies a non-linearity between layers, which is what stops the collapse. It is a small operation with an outsized structural role: without it there is no such thing as a deep network.
