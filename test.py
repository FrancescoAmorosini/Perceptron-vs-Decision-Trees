import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
import random

def plot_learning_curve(estimator, x, y, iterations = 10):
    from sklearn.model_selection import PredefinedSplit

    #Split the data
    my_test_fold = np.zeros(len(x))
    test_size = 10000
    train_folds = 6
    
    for i in range(len(x)-test_size):
        my_test_fold[i] = -1

    cv = PredefinedSplit(my_test_fold)

    train_acc = []
    test_acc = []

    train_indexes = list(range(len(x) - test_size))
    test_indexes = list(range(len(train_indexes), len(train_indexes) + test_size))

    for j in range(iterations):
        #Shuffle training set
        random.shuffle(train_indexes)
        actual_indexes = np.append(train_indexes, test_indexes)
        x_iter = x[actual_indexes]
        y_iter = y[actual_indexes]

        #Calculate accuracy scores
        train_sizes, train_scores, test_scores = learning_curve(estimator, x_iter, y_iter, 
        cv= cv, train_sizes=np.logspace(-3,0,train_folds), scoring='accuracy', verbose=1)
        train_acc.append(train_scores)
        test_acc.append(test_scores)

    #Calculate mean and standard deviation for each train size
    train_scores_mean = np.mean(train_acc, axis=0)
    train_scores_std = np.std(train_acc, axis=0)
    test_scores_mean = np.mean(test_acc, axis=0)
    test_scores_std = np.std(test_acc, axis=0)

    plt.fill_between(train_sizes, np.reshape((train_scores_mean - train_scores_std),train_folds),
                     np.reshape((train_scores_mean + train_scores_std),train_folds), alpha=0.1,
                     color="red")
    plt.fill_between(train_sizes, np.reshape((test_scores_mean - test_scores_std),train_folds),
                     np.reshape((test_scores_mean + test_scores_std),train_folds), alpha=0.1, color="blue")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="red",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="blue",
             label="Test score")
    plot_results()
    
def plot_results():
    plt.xscale("log")
    plt.ylabel('Accuracy Score')
    plt.xlabel('Training Set Size')
    plt.legend(loc='best')
    plt.title("Learning Curves")
    plt.grid(axis='y',linewidth=0.3)
    plt.show()
