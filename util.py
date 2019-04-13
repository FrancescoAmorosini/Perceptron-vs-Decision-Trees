
def load_data():
    import os
    if not os.path.isdir('./data'):
        import git
        import shutil
        import tempfile

        # Create temporary dir
        t = tempfile.mkdtemp()
        # Clone dataset into temporary dir
        git.Repo.clone_from('https://github.com/zalandoresearch/fashion-mnist.git', t, branch='master', depth=1)
        # Copy desired files from temporary dir into data dir
        os.mkdir('./data')
        if not os.path.isdir('./data/fashion'):
            shutil.move(os.path.join(t, 'data/fashion'), './data')
        if not os.path.isfile('./data/mnist_reader.py'):
            shutil.move(os.path.join(t, 'utils/mnist_reader.py'), './data')
    
    #Load dataset from data folder
    from data.mnist_reader import load_mnist

    x_train, y_train = load_mnist('data/fashion', kind='train')
    x_test, y_test = load_mnist('data/fashion', kind='t10k')

    return x_train, y_train, x_test, y_test

def scale_data(x_train, x_test):
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings('ignore') 

    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    return x_train_std, x_test_std

def draw_tree(tree):
    #Generates a .png of the decision tree
    import pydotplus
    import collections

    categories = ['T-shirt', 'Trouser', 'Pullover', 'Dress','Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    dot_data = tree.export_graphviz(tree, out_file=None,class_names=categories,
    filled=False, rounded=True, special_characters=True, leaves_parallel=False)
    
    graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
    graph.write_png("tree.png")

def draw_image(sprite_array):
    import matplotlib.pyplot as plt
    import draw_sprite
    plt.imshow(draw_sprite.get_sprite_image(sprite_array),'gray')
    plt.show()
