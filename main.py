from sklearn.linear_model import Perceptron
from sklearn import tree
import util
import test
import numpy as np

def main():
    #Import dataset
    x_train, y_train, x_test, y_test = util.load_data()
    
    #Sample standardization
    x_train_std, x_test_std = util.scale_data(x_train, x_test)

    #Create Models using Sklearn
    decision_tree = tree.DecisionTreeClassifier(min_impurity_decrease= 1e-3)
    perceptron = Perceptron(tol=1e-4)
    iters = 30

    #Draw learning curves
    test.plot_learning_curve(perceptron, np.concatenate((x_train_std, x_test_std)), np.concatenate((y_train, y_test)), iters)
    test.plot_learning_curve(decision_tree, np.concatenate((x_train_std, x_test_std)), np.concatenate((y_train, y_test)), iters)

    ''' It's possible to visualize data sprites using util.draw_image(x_train[<index>])
        It's also possible to visualize the decision tree using util.draw_tree(decision_tree)'''

if __name__ == '__main__':
	main()


    