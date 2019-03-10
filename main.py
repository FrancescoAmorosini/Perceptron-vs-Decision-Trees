from FashionMNIST.utils.mnist_reader import load_mnist
import test_perceptron
import draw_data



def main():
    x_train, y_train = load_mnist('FashionMNIST/data/fashion', kind='train')
    x_test, y_test = load_mnist('FashionMNIST/data/fashion', kind='t10k')
    #draw_data.draw_sprite(x_train[0])

    #test_perceptron.sklearning_curve(x_train, y_train, x_test, y_test, 10)
    test_perceptron.learning_curve(x_train, y_train, x_test, y_test,5)

    test_perceptron.plot_results()
if __name__ == '__main__':
	main()

    