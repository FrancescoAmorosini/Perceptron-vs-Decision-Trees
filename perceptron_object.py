
import numpy as np
import matplotlib.pyplot as plt
import random as rd

class PerceptronMNIST(object):
	#Optimal parameters: 25 epochs, 0.3 alpha
	def __init__(self, data, epochs = 20, alpha = .001):
		self.categories = ['T-shirt', 'Trouser', 'Pullover', 'Dress','Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
		self.epochs = epochs
		self.alpha = alpha
		self.x_train, self.y_train, self.x_test, self.y_test = data
		self.train_labels = self.vectorize_labels(self.y_train)
		self.test_labels = self.vectorize_labels(self.y_test)
		#self.normalized_train = [x / 255 for x in self.x_train]
		#self.normalized_test = [x / 255 for x in self.x_test]
		self.weights, self.bias = self.train_all()
		self.predictions = self.predict_all()
		self.accuracy = self.test_all()

	def train_all(self):
		w = []
		b = []
		for category in range(len(self.categories)):
			w.append([])
			b.append([])
			w[category],b[category] = self.perceptron_train(category, self.epochs)
		return w,b

	def perceptron_train(self, category, max_iter = 10):
		w = np.zeros(np.shape(self.x_train)[1])
		#w = np.random.rand(np.shape(self.x_train)[1])
		b = 0
		iter = 0
		while iter < max_iter:
			for i in range(len(self.x_train)):
				x = self.x_train[i]
				y = self.train_labels[i][category]
				if y * (np.dot(x, w) + b) <= 0:
					delta = (np.multiply(y, x))*self.alpha
					w = np.add(w, delta)
					b += y
			iter +=1
		return w, b

	def predict_all(self):
		pred=[]
		for i in range(len(self.x_test)):
			pred.append([])
			for j in range(len(self.categories)):
				pred[i].append(self.perceptron_predict(self.weights[j],self.bias[j],self.x_test[i]))
		return pred

	def perceptron_predict(self, w, b, sample):
		return np.sign(np.dot(sample, w) + b)

	def vectorize_labels(self, y):
		vector = []
		for j in range(len(y)):
			vector.append([])
			for i in range(len(self.categories)):
				vector[j].append(0)
				if i == y[j]: vector[j][i]+=1
				else: vector[j][i]-=1
		return vector
	
	def test_all(self):
		n_correct = n_wrong =  0

		for i in range(len(self.x_test)):
			if max(self.predictions[i])== 1 and self.predictions[i].index(1) == self.y_test[i]:
				n_correct+=1
			else: n_wrong +=1
		
		accuracy = float((n_correct) / (len(self.x_test)))
		return accuracy
	
	'''def test_all(self):
		tp = tn = fp = fn = 0 

		for i in range(len(self.x_test)):
			for j in range (len(self.categories)):
				if self.predictions[i][j]== 1 and self.test_labels[i][j] == 1:
					tp +=1
				elif self.predictions[i][j]== -1 and self.test_labels[i][j] == -1:
					tn +=1
				elif self.predictions[i][j]== 1 and self.test_labels[i][j] == -1:
					fp +=1
				else: fn +=1
		
		accuracy = float((tp+tn) / (tp+tn+fp+fn))
		return accuracy '''
