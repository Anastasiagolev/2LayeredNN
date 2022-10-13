# 2 Layer Nueral Network in numpy
#Builing a simple nueral network
#The goal of this project is to implement a 3-input XNOR gate with inputs X1, X2, X3 and have Y1 to be the result.

import numpy as np

#X is the input of our 3 input XOR gate. This will set up the inputs of the nueral network
X = np.array(([0,0,0], [0,0,1], [0,1,1],[1, 0, 0], [1,0,1],[1,1,0],[1,1,1]), dtype = float)
#y is the output of our neural network
y = np.array(([1], [0], [0], [0], [0], [0], [1]), dtype = float)

#we will have variable xPredicted to represent the value we want to predict
xPredicted = np.array(([0,0,1]), dtype = float)

#maximum of X input array
X = X/np.amax(X, axis = 0)

#maximum of xPredicted
xPredicted = xPredicted/np.amax(xPredicted, axis = 0)

#set us our Loss file for graphing
lossFile = open("SumSquaredLossList.csv", "w")

#start Nueral_Network class definition
class Neural_Network(object):
    def __init__(self):
        #parameters for input, output and hidden layers
        self.inputLayerSize = 3
        self.outputLayerSize = 1
        self.hiddenLayerSize = 4

        #now we build the weights of each layer. We set them to random values

        #the matrix for input to hidden layer will be 3x4
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        #the matrix for hidden to output layer will be 4x1
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    #This NN will be feed-forward input. Layers will filter and process info from left to right
    def feedForward(self, X):
        #we will perform the dot product of X and the first set of 3x4 weights
        self.z = np.dot(X, self.W1)

        #we will use the Sigmoid activation function
        self.z2 = self.activationSigmoid(self.z)

        #perform dot product of the hidden layer and the second set of 4x1 weights
        self.z3 = np.dot(self.z2, self.W2)

        #final activation function
        o = self.activationSigmoid(self.z3)

        return o

    def backPropagate(self, X, y, o):
        #we will backward propagate through the network

        #calculate the error in output
        self.o_error = y - o

        #apply the derivative of our Sigmoid function to the error
        self.o_delta = self.o_error*self.activationSigmoidPrime(o)

        #calculate z2 error by performing the dot product with W2 transposed
        #this will tell us how much our hidden layer weights contributed to output error
        self.z2_error = self.o_delta.dot(self.W2.T)

        #apply the derivative of Sigmoid function to z2 error
        self.z2_delta = self.z2_error*self.activationSigmoidPrime(self.z2)

        #the we adjust the input to hidden layer weights
        self.W1 += X.T.dot(self.z2_delta)
        #and the hidden to output layer weights
        self.W2 += self.z2.T.dot(self.o_delta)

    def trainNetwork(self, X, y):
        #we feed forward the loop
        o = self.feedForward(X)

        #and backward propagate the values (ie the feedback)
        self.backPropagate(X, y, o)

    #function definition for our Sigmoid activation function
    def activationSigmoid(self, s):
        return 1/(1+np.exp(-s))

    #function definition for the first derivative of activationSigmoid
    def activationSigmoidPrime(self, s):
        return s*(1-s)

    def saveSumSquaredLossList(self, i, error):
        lossFile.write(str(i) + ", "+str(error.tolist())+'\n')

    def saveWeights(self):
        np.savetxt("weightsLayer1.txt", self.W1, fmt = "%s")
        np.savetxt("weightsLayer2.txt", self.W2, fmt="%s")

    def predictOutput(self):
        print("Predicted XOR output data is based on trained weights: ")
        print("Expected (X1-X3): \n"+ str(xPredicted))
        print("Output (Y1): \n "+str(self.feedForward(xPredicted)))

myNN = Neural_Network()
epochs = 1000
#we will train the NN for 1000 runs
for i in range(epochs):
    print("Epoch # " + str(i) + "\n")
    print("Network Input : \n" + str(X))
    print("Expected Output of XOR Gate Neural Network: \n" + str(y))
    print("Actual  Output from XOR Gate Neural Network: \n" +str(myNN.feedForward(X)))
    # mean sum squared loss
    Loss = np.mean(np.square(y - myNN.feedForward(X)))
    myNN.saveSumSquaredLossList(i, Loss)
    print("Sum Squared Loss: \n" + str(Loss))
    print("\n")
    myNN.trainNetwork(X, y)

myNN.saveWeights()
myNN.predictOutput()
