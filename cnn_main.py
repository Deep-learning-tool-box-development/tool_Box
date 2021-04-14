
from dataset import import_data
from pso import PSO
from cnn import CNN
from sa import SA


def run_cnn_pso(path_to_data, var_size, discrete_candidate):
    """
    Main function for the CNN and PSO.
    :param path_to_data: string, Folder of the data files
    :param var_size: list, upper and under boundaries of all variables
    :param discrete_candidate: list, list of discrete params, convolution layer and batch size
    :return: None
    """
    # create the CNN model
    x_train, x_test, y_train, y_test = import_data(path_to_data, model='CNN')
    cnn = CNN(x_train, y_train, x_test, y_test,
              discrete_candidate=discrete_candidate,
              optimization=True,
              epoch=6)
    pso = PSO(cnn.cnn_get_score, 5, 5, var_size, candidate=discrete_candidate, net="CNN")
    pso.run()


def run_cnn_sa(path_to_data, var_size, discrete_candidate):
    """
    Main function for the CNN and SA.
    :param path_to_data: string, Folder of the data files
    :param var_size: list, upper and under boundaries of all variables
    :param discrete_candidate: list, list of discrete params, convolution layer and batch size
    :return: None
    """
    x_train, x_test, y_train, y_test = import_data(path_to_data, model='CNN')
    cnn = CNN(x_train, y_train, x_test, y_test,
              discrete_candidate=discrete_candidate,
              optimization=True,
              epoch=6)
    sa = SA(cnn.cnn_get_score, 200, 10, 0.9, var_size, candidate=discrete_candidate, net="CNN")
    sa.run()


if __name__ == '__main__':
    """
    Main function to call the selected model and optimizer
    """
    # Main function
    option = 'PSO'
    path = './Dataset/'
    # below should get from config
    do = [0.3, 0.8]  # dropout
    lr = [0.0001, 0.02]  # learning rate
    # 6 candidate of batch size
    conv_candidate = [4, 6, 8]  # convolution
    bs_candidate = [1, 16, 32, 64, 128, 256]  # batch size
    discrete_candi = [bs_candidate, conv_candidate]
    # 3 candidate of number of the convolution layers
    var_size = [do, lr, [0, 1], [0, 1]]  # lower and upper bounds of all params
    if option == 'PSO':
        run_cnn_pso(path, var_size, discrete_candi)

    elif option == 'SA':
        run_cnn_sa(path, var_size, discrete_candi)

