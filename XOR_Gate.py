import numpy as np
from matplotlib import pyplot as plt

def ReLU(x):
    return np.maximum(0, x)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initializeParameters(inputFeatures, neuronsInHiddenLayers, outputFeatures):
    W1 = np.random.randn(neuronsInHiddenLayers, inputFeatures) * np.sqrt(2 / inputFeatures)
    W2 = np.random.randn(outputFeatures, neuronsInHiddenLayers) * np.sqrt(2 / neuronsInHiddenLayers)
    b1 = np.zeros((neuronsInHiddenLayers, 1))
    b2 = np.zeros((outputFeatures, 1))

    parameters = {"W1" : W1, "b1" : b1, "W2" : W2, "b2" : b2}

    return parameters

def forwardPropogation(X, Y, parameters):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2)
    epsilon = 1e-8
    A2 = np.clip(A2, epsilon, 1 - epsilon)
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = -np.sum(logprobs)/m
    return cost, cache, A2

def backPropogation(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2) = cache
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis = 1, keepdims = True)/m

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / m

    gradients = {"dZ2" : dZ2, "dW2" : dW2, "db2" : db2, "dZ1" : dZ1, "dW1" : dW1, "db1" : db1}

    return gradients


def UpdateParameters(parameters, gradients, learningRate):
    parameters["W1"] = parameters["W1"] - learningRate * gradients["dW1"]

    parameters["W2"] = parameters["W2"] - learningRate * gradients["dW2"]

    parameters["b1"] = parameters["b1"] - learningRate * gradients["db1"]

    parameters["b2"] = parameters["b2"] - learningRate * gradients["db2"]

    return parameters

def plot_decision_boundary(parameters, X, Y):
    x1 = np.linspace(-0.5, 1.5, 100)
    x2 = np.linspace(-0.5, 1.5, 100)

    xx1, xx2 = np.meshgrid(x1, x2)
    grid = np.c_[xx1.ravel(), xx2.ravel()].T
    
    dummy_Y = np.zeros((1, grid.shape[1]))
    _, _, A2 = forwardPropogation(grid, dummy_Y, parameters)

    Z = A2.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, levels = [0, 0.5, 1], alpha = 0.3)
    plt.contour(xx1, xx2, Z, levels = [0.5], colors = 'red', linewidths = 2)
    plt.scatter(X[0, :], X[1, :], c = Y.flatten(), edgecolors = 'k')


# def plot_hidden_neurons(parameters, X, Y):
#     W1 = parameters["W1"]
#     b1 = parameters["b1"]

#     x1 = np.linspace(-0.5, 1.5, 200)
#     x2 = np.linspace(-0.5, 1.5, 200)

#     xx1, xx2 = np.meshgrid(x1, x2)
#     grid = np.c_[xx1.ravel(), xx2.ravel()].T

#     Z1 = np.dot(W1, grid) + b1  # (neurons, points)

#     plt.figure()

#     for i in range(Z1.shape[0]):
#         Zi = Z1[i, :].reshape(xx1.shape)

#         # Plot where neuron "switches"
#         plt.contour(xx1, xx2, Zi, levels=[0], linewidths=2)

#     plt.scatter(X[0, :], X[1, :], c=Y.flatten(), edgecolors='k')
#     plt.title("Hidden Layer Boundaries")
#     plt.show()




X = np.array([[0, 0, 1, 1],
               [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])



neuronsInHiddenLayers = 4
inputFeatures = X.shape[0]
outputFeatures = Y.shape[0]

parameters = initializeParameters(inputFeatures, neuronsInHiddenLayers, outputFeatures)
epoch = 10000
learningRate = 0.5
losses = np.zeros((epoch, 1))


dummy_Y = np.zeros((1, X.shape[1]))

for i in range(epoch):
    losses[i, 0], cache, A2 = forwardPropogation(X, Y, parameters)
    gradients = backPropogation(X, Y, cache)
    parameters = UpdateParameters(parameters, gradients, learningRate)
    if(i % 2000 == 0):
        plt.figure()
        plot_decision_boundary(parameters, X, Y)
plt.title(f"Boundary Evolution{i}")
plt.show()
plt.figure()
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss Value")

X = np.array([[1, 1, 0, 0],
               [0, 1, 0, 1]])
_, _ , A2 = forwardPropogation(X, dummy_Y, parameters)
prediction = (A2 > 0.5) * 1.0
print(A2)
# plot_hidden_neurons(parameters, X, Y)
print(prediction)
plt.show()