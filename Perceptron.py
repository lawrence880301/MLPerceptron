import numpy as np

class Perceptron(object):

    def __init__(self, no_of_inputs, threshold, learning_rate):
        self.threshold = threshold #epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1) #input add bias
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                if prediction > label:
                    self.weights[1:] -= self.learning_rate * inputs
                    self.weights[0] -= self.learning_rate 
                if prediction < label:
                    self.weights[1:] += self.learning_rate * inputs
                    self.weights[0] += self.learning_rate 

