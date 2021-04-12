import numpy as np
from dataset import import_data
from pso import PSO
from sa import SA
from svm import SVM_Model
import math


def run_svm_pso(path_to_data, var_size):
    """
    Main function for the SVM and PSO.
    :param path_to_data: string, Folder of the data files
    :param var_size: list, upper and under boundaries of all variables
    :return: None
    """
    x_train, x_test, y_train, y_test = import_data(path_to_data, model='SVM')
    svm = SVM_Model(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, optimization=True)
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
    svm = SVM_Model(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, optimization=True)
    sa = SA(svm.get_score, 200, 10, 0.9, var_size, net="SVM")
    sa.run()
    

if __name__ == '__main__':
    """
    Main function to call the selected model and optimizer
    """
    # Main function
    option = 'Test' # switch between "Test", "SA", "PSO"
    path = './dataset/'
    # below should get from config
    C = [5, 100]
    gamma = [0, 1]
    var_size=np.append([np.array(C)/C[1]],[np.array(gamma)/gamma[1]],axis=0) 

    if option == 'PSO':
        run_svm_pso(path, var_size)

    elif option == 'SA':
        run_svm_sa(path, var_size)
    # Below test only    
    elif option == 'Test':
        x_train, x_test, y_train, y_test = import_data(path, model='SVM')
        svm = SVM_Model(x_train=x_train, y_train=y_train, x_test=x_test, y_test = y_test, optimization=False)
        params = [50, 0.026]
        svm.train_svm(params)