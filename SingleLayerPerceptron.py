from random import seed
from random import randrange
from DataPreprocessor import Datapreprocessor
# Perceptron Algorithm on the Sonar Dataset
from random import seed
from random import randrange

class SingleLayerPerceptron():
	def __init__(self) -> None:
		self.weights = [0]

	# Calculate accuracy percentage
	def accuracy_metric(self, actual, predicted):
		correct = 0
		for i in range(len(actual)):
			if actual[i] == predicted[i]:
				correct += 1
		return correct / float(len(actual)) * 100.0

	# Evaluate an algorithm using a cross validation split
	def evaluate_algorithm(self, dataset, algorithm, *args):
		train_set, test_set = Datapreprocessor.train_test_split(dataset,2/3)
		scores = list()
		if algorithm == "perceptron":
			predicted = self.perceptron(train_set, test_set, *args)
		actual = [row[-1] for row in test_set]
		accuracy = self.accuracy_metric(actual, predicted)
		scores.append(accuracy)
		return scores

	# Make a prediction with weights
	def predict(self, row):
		activation = self.weights[0]
		for i in range(len(row)-1):
			activation += self.weights[i + 1] * row[i]
		return self.hard_limit(activation)

	def hard_limit(self, x):
		return 1.0 if x >= 0.0 else 0.0


	def train_weights(self, train, l_rate, n_epoch):
		self.weights = [0.0 for i in range(len(train[0]))] #weights including input weight and bias weight
		for epoch in range(n_epoch):
			for row in train:
				prediction = self.predict(row)
				error = row[-1] - prediction
				#update bias
				self.weights[0] = self.weights[0] + l_rate * error
				#update weights
				for i in range(len(row)-1):
					self.weights[i + 1] = self.weights[i + 1] + l_rate * error * row[i]
		

	# Perceptron Algorithm With Stochastic Gradient Descent
	def perceptron(self, train, test, l_rate, n_epoch):
		predictions = list()
		self.train_weights(train, l_rate, n_epoch)
		for row in test:
			prediction = self.predict(row)
			predictions.append(prediction)
		return predictions


# load and prepare data
filepath = '/Users/lawrence/Documents/類神經網路/homework1/hw_yuan/hw1/NN_HW1_DataSet/NN_HW1_DataSet/2Ccircle1.txt'
dataset = Datapreprocessor.readfile(filepath)
dataset = Datapreprocessor.text_to_numlist(dataset)
n_folds = 3
l_rate = 0.01
n_epoch = 5000
model = SingleLayerPerceptron()
scores = model.evaluate_algorithm(dataset, 'perceptron', l_rate, n_epoch)
print('Scores: %s' % scores)
print(model.weights)
