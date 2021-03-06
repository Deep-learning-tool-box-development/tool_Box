from dataset import import_data
from pso import PSO
from dbn import DBN
from sa import SA


def run_dbn_pso(path_to_data, var_size):
    """
    Main function for the DBN and PSO.

    :param path_to_data: string, Folder of the data files
    :param var_size: list, upper and under boundaries of all variables
    :return: None
    """
    # create the DBN model
    x_train, x_test, y_train, y_test = import_data(path_to_data, model='DBN')
    dbn = DBN(train_data=x_train, targets=y_train,
              batch_size=64,
              rbm_layers = [200,50],
              outdir=None,
              logdir=None,
              optimization=True)
    pso = PSO(dbn.dbn_get_score, 4, 10, var_size, net = "DBN")
    pso.run()


def run_dbn_sa(path_to_data, var_size):
    """
    Main function for the DBN and SA.

    :param path_to_data: string, Folder of the data files
    :param var_size: list, upper and under boundaries of all variables
    :return: None
    """
    x_train, x_test, y_train, y_test = import_data(path_to_data, model='DBN')
    
    dbn = DBN(train_data=x_train, targets=y_train,
              batch_size=64,
              rbm_layers = [200,50],
              outdir=None,
              logdir=None,
              optimization=True)
    sa = SA(dbn.dbn_get_score, 500, 1, 0.9, 30, var_size, net="DBN")
    # objective, initial_temp, final_temp, alpha, max_iter, var_size, candidate, net
    sa.run()


if __name__ == '__main__':
    """
    Main function to call the selected model and optimizer
    """
    # Main function
    option = 'PSO'
    path = './Dataset/'
    
    var_size = [[0.5, 0.8], [1e-4, 1e-3], [1e-4, 1e-3]]  # var_size = [Dropout,LearningRate_RBM,LearningRate_nn] 
    if option == 'PSO':
        run_dbn_pso(path, var_size)

    elif option == 'SA':
        run_dbn_sa(path, var_size)


