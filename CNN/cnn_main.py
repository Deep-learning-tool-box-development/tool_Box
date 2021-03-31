
from dataset import import_data
from pso import PSO
from cnn import CNN
from sa import SA


def run_cnn_pso(path_to_data, var_size):
    """
    Main function for the CNN and PSO.

    :param path_to_data: string, Folder of the data files
    :return: None
    """
    # create the CNN model
    x_train, x_test, y_train, y_test = import_data(path_to_data, model='CNN')
    cnn = CNN(x_train, y_train, x_test, y_test)
    pso = PSO(cnn.cnn_get_score, 5, 10, var_size)
    pso.run()


def run_cnn_sa(path_to_data):
    x_train, x_test, y_train, y_test = import_data(path_to_data, model='CNN')
    cnn = CNN(x_train, y_train, x_test, y_test)
    var_size = [[0.5, 0.8], [0.001, 0.01], [1, 3.5]]
    sa = SA(cnn.cnn_get_score, 90, 0.1, 0.01, var_size)
    sa.run()


if __name__ == '__main__':
    """
    Main function to call the selected model and optimizer
    """
    # Main function
    option = 'PSO'
    path = 'PATH TO DATASET'
    # below should get from config
    do = [0.5, 0.8]
    lr = [0.0001, 0.01]
    # 6 candidate of batch size
    bs = [0, 5.99]
    # 3 candidate of number of the convolution layers
    num_conv = [0, 2.99]
    var_size = [do, lr, bs, num_conv]
    if option == 'PSO':
        run_cnn_pso(path, var_size)

    elif option == 'SA':
        run_cnn_sa(path)