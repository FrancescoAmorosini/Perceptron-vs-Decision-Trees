import perceptron_object
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np

def learning_curve(x_train, y_train, x_test, y_test, iterations = 10):
    sizes = []
    accuracies = []
    for iter in range(1,iterations+1):
        train_dim = int((len(x_train)/iterations)*iter)
        x = x_train[:train_dim]
        y = y_train[:train_dim]
        data = (x, y, x_test, y_test)
        perceptron = perceptron_object.PerceptronMNIST(data)
        sizes.append(float(int((len(x_train)/iterations)*iter*100)/(len(x_train))))
        accuracies.append(perceptron.accuracy * 100)
        print ("Accuracy: ", accuracies[iter-1],
         "%, Training set: ", sizes[iter-1], "%")
    plt.scatter(sizes, accuracies, label="My Perceptron")

def sklearn_perceptron(x_train, y_train):
    clf = Perceptron(tol=1e-3, max_iter=20, alpha=.001, random_state=0)
    clf.fit(x_train,y_train)
    return clf

def sklearning_curve(x_train, y_train, x_test, y_test, iterations = 10):
    sizes = []
    accuracies = []
    for iter in range(1,iterations+1):
        train_dim = int((len(x_train)/iterations)*iter)
        x = x_train[:train_dim]
        y = y_train[:train_dim]
        perceptron = sklearn_perceptron(x, y)
        sizes.append(float(int((len(x_train)/iterations)*iter*100)/(len(x_train))))
        accuracies.append(perceptron.score(x_test,y_test)*100)
        print ("Accuracy: ", accuracies[iter-1],
         "%, Training set: ", sizes[iter-1], "%")
    plt.scatter(sizes, accuracies, label="Sklearn Perceptron")

def plot_results():
        axes = plt.gca()
        axes.yaxis.set_ticks(np.arange(0,100,10))
        plt.ylabel('% Accuracy')
        plt.xlabel('% Training Set')
        plt.legend(loc='lower right')
        plt.title("Accuracy Results")
        plt.grid(axis='y',linewidth=0.3)
        plt.show()
