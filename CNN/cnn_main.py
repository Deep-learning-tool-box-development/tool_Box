
from dataset import import_data
from pso import PSO
from cnn import CNN
from sa import SA


def run_cnn_pso(path_to_data, var_size):
    """
    Main function for the CNN and PSO.
    :param var_size: list, upper and under boundaries of all variables
    :param path_to_data: string, Folder of the data files
    :return: None
    """
    # create the CNN model
    x_train, x_test, y_train, y_test = import_data(path_to_data, model='CNN')
    cnn = CNN(x_train, y_train, x_test, y_test)
    pso = PSO(cnn.cnn_get_score, 5, 10, var_size)
    pso.run()


def run_cnn_sa(path_to_data, varsize):
    """
    Main function for the CNN and SA.
    :param var_size: list, upper and under boundaries of all variables
    :param path_to_data: string, Folder of the data files
    :return: None
    """
    x_train, x_test, y_train, y_test = import_data(path_to_data, model='CNN')
    cnn = CNN(x_train, y_train, x_test, y_test)
    sa = SA(cnn.cnn_get_score, 200, 10, 0.9, var_size, net="CNN")
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
    # All discrete parameters use var_size of [0, 1]
    bs = [0, 1]
    num_conv = [0, 1]
    var_size = [do, lr, bs, num_conv]
    if option == 'PSO':
        run_cnn_pso(path, var_size)

    elif option == 'SA':
        run_cnn_sa(path, var_size)
