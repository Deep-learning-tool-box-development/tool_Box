import numpy as np
from dataset import import_data
from pso import PSO
from sa import SA
from svm import SVM_Model


def run_svm_pso(path_to_data, var_size):
    """
    Main function for the SVM and PSO.

    :param path_to_data: string, Folder of the data files
    :param var_size: list, upper and under boundaries of all variables
    :return: None
    """
    x_train, x_test, y_train, y_test = import_data(path_to_data, model='SVM')
    svm = SVM_Model(x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test,
                    optimization=True)
    pso = PSO(svm.get_score, 2, 5, var_size, net="SVM")
    pso.run()


def run_svm_sa(path_to_data, var_size):
    """
    Main function for the SVM and SA.

    :param path_to_data: string, Folder of the data files
    :param var_size: list, upper and under boundaries of all variables
    :return: None
    """
    x_train, x_test, y_train, y_test = import_data(path_to_data, model='SVM')
    svm = SVM_Model(x_train=x_train, y_train=y_train,
                    x_test=x_test, y_test=y_test,
                    optimization=True)
    sa = SA(svm.get_score, 500, 1, 0.9, 30,  var_size, net="SVM")
    # objective, initial_temp, final_temp, alpha, max_iter, var_size, net
    sa.run()


if __name__ == '__main__':
    """
    Main function to call the selected model and optimizer
    """
    # Main function
    option = "SA"  # switch between "SA", "PSO"
    path = './dataset/'
    # below should get from config

    var_size = [[15, 100], [0.001, 0.01]]  # var_size = [C,gamma]

    if option == 'PSO':
        run_svm_pso(path, var_size)

    elif option == 'SA':
        run_svm_sa(path, var_size)
