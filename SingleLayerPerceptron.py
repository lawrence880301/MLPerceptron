from random import seed
from random import randrange
from DataPreprocessor import Datapreprocessor
# Perceptron Algorithm on the Sonar Dataset
from random import seed
from random import randrange

class SingleLayerPerceptron():
	def __init__(self, l_rate, epoch):
		self.weights = [0]
		self.l_rate = l_rate
		self.n_epoch = epoch
		self.train_set = []
		self.test_set = []
	# Calculate accuracy percentage
	def accuracy_metric(self, actual, predicted):
		correct = 0
		for i in range(len(actual)):
			if actual[i] == predicted[i]:
				correct += 1
		return correct / float(len(actual)) * 100.0

	# Evaluate an algorithm using a cross validation split
	def evaluate_algorithm(self, dataset, algorithm, *args):
		self.train_set, self.test_set = Datapreprocessor.train_test_split(dataset,2/3)
		scores = list()
		if algorithm == "perceptron":
			predicted = self.perceptron(self.train_set, self.test_set, *args)
		actual = [row[-1] for row in self.test_set]
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
		return 1.0 if x >= 0.0 else -1.0


	def train_weights(self, train):
		self.weights = [0.0 for i in range(len(train[0]))] #weights including input weight and bias weight
		for epoch in range(self.n_epoch):
			for row in train:
				prediction = self.predict(row)
				if prediction > row[-1]:
					#update bias
					self.weights[0] -= self.l_rate
					#update weights
					for i in range(len(row)-1):
						self.weights[i + 1] -= self.weights[i + 1] + self.l_rate * row[i]
				if prediction < row[-1]:
					#update bias
					self.weights[0] += self.l_rate
					#update weights
					for i in range(len(row)-1):
						self.weights[i + 1] += self.weights[i + 1] + self.l_rate * row[i]

		

	# Perceptron Algorithm With Stochastic Gradient Descent
	def perceptron(self, train, test):
		predictions = list()
		modified_train_set = Datapreprocessor.label_preprocess(train)
		modified_test_set = Datapreprocessor.label_preprocess(test)
		self.train_weights(modified_train_set)
		for row in modified_test_set:
			prediction = self.predict(row)
			predictions.append(prediction)
		return predictions



