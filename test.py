import numpy as np
from DataPreprocessor import Datapreprocessor

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

# def readfile(url):
#     read = open(url)
#     file = read.readlines()
#     read.close()
#     return file

# def prepare_data():
#     dataset = Datapreprocessor.readfile(dataset_url)
#     dataset = Datapreprocessor.text_to_numlist(dataset)
#     return dataset

# dataset_url = "/Users/lawrence/Documents/類神經網路/homework1/ping/Perceptron-main/DataSet/2CS.txt"
# dataset = readfile(dataset_url)
# dataset = Datapreprocessor.text_to_numlist(dataset)
# dataset = Datapreprocessor.label_preprocess(dataset)
# train_data, test_data = Datapreprocessor.train_test_split(dataset, 2/3)
# train_x, train_y = Datapreprocessor.feature_label_split(train_data)
# test_x, test_y = Datapreprocessor.feature_label_split(test_data)
# model = Perceptron(len(dataset[1])-1)
# model.train(np.array(train_x), np.array(train_y))

# def accuracy_metric(actual, predicted):
# 	correct = 0
# 	for i in range(len(actual)):
# 		if actual[i] == predicted[i]:
# 			correct += 1
# 	return correct / float(len(actual)) * 100.0

# predict_list = []
# for row in test_x:
# 	predict = model.predict(row)
# 	predict_list.append(predict)

# print(accuracy_metric(test_y, predict_list))
