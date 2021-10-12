import numpy as np
class MLPerceptron():
    def __init__(self, n_hiddenlayers, n_neuron, n_input, n_output, lr) -> None:
        self.lr = lr
        self.layers = []
        self.layers.append(Layer(n_input, n_neuron))
        [self.layers.append(Layer(n_neuron, n_neuron)) for i in range(n_hiddenlayers)]
        self.layers.append(Layer(n_neuron, n_output))

    def forward(self, x):
        for layer in self.layers:
            x = layer.activation(x)
        return x

    def backward(self, x, y):
        out = self.forward(x)
        #calculate delta for each layer
        for i in reversed(range(len(self.layers))):
            # last layer
            if i == len(self.layers) - 1:
                self.layers[i].error = y - out
                self.layers[i].delta = self.layers[i].error * self.layers[i].activation_der(out)
            # hidden layers
            else :
                self.layers[i].delta = np.dot(self.layers[i].activation_der(self.layers[i].output),
                                    np.dot(self.layers[i+1].weights, self.layers[i+1].delta))
        #modify weight for each layer
        for i, layer in enumerate(self.layers):
            layer = self.layers[i]
            output = np.atleast_2d(x if i == 0 else self.layers[i - 1].output)
            layer.weights = layer.weights + layer.delta * output.T * self.lr

    def train(self, x, y, n_iter):
        losses =[]
        for i in range(n_iter):
            for xi, yi in zip(x, y):
                self.backward(xi, yi)
            loss = np.sum((y-self.forward(x))**2)
            losses.append(loss)
        return losses
    
    def predict(self, x):
        outputs = self.forward(x)
        return np.argmax(outputs)


class Layer:
    def __init__(self, n_input, n_neuron):
        
        self.weights = np.random.rand(n_input, n_neuron)
        self.bias = np.ones(n_neuron)
        self.delta = 0
        ##error only used in the output layer
        self.error = 0
        self.weighted_sum = 0
        
    def cal_weighted_sum(self, x):
        self.weighted_sum = np.dot(x, self.weights) + self.bias
        return self.weighted_sum
    
    def activation(self, x):
        self.output = 1 / (1 + np.exp(-self.cal_weighted_sum(x)))
        return self.output
    
    
    def activation_der(self, out):
        return out*(1-out)
