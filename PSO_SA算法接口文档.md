# PSO / SA optimization algorithms deployment



concrete CNN，PSO，SA, DBN interface definition to be seen in basic algorithms description.

# cnn_main

cnn_main function is to use pso and sa to optimize parameters of cnn:

to switch between "PSO" and "SA", user only need to change `option` 

- if option == "PSO" -- run_cnn_pso: input are path to data, upper and under boundaries of all variables, and discrete candidates

  the parameters of pso are set as: cnn.cnn_get_score, 5, 5, var_size, candidate=discrete_candidate, net="CNN"

- if option == "SA" -- run_cnn_sa: input are path to data, upper and under boundaries of all variables, and discrete candidates

  he parameters of pso are set as: cnn.cnn_get_score, 200, 10, 0.9, var_size, candidate=discrete_candidate, net="CNN"

variables to be optimized are dropout, learning rate, convolution and batch size:

-  dropout, in range [0.3, 0.8] 
- learning rate in range [0.0001, 0.02]
- convolution as discrete parameter choose among [4, 6, 8] 
- batch size as discrete parameter choose among [1, 16, 32, 64, 128, 256] 

```python
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
    option = 'PSO' # choose between "PSO" and "SA"
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

```

# dbn_main

dbn_main function is to use pso and sa to optimize parameters of dbn:

to switch between "PSO" and "SA", user only need to change `option` 

- if option == "PSO" -- run_dbn_pso: input are path to data, upper and under boundaries of all variables

  the parameters of pso are set as: dbn.dbn_get_score, 4, 10, var_size, net = "DBN"

- if option == "SA" -- run_dbn_sa: input are path to data, upper and under boundaries of all variables

  he parameters of pso are set as: dbn.dbn_get_score, 200, 10, 0.9, var_size, net="DBN"

variables to be optimized are dropout, learning rate_RBM, learning rate_nn:

-  dropout, in range [0.5, 0.8] 
-  learning rate_RBM, in range [1e-4, 1e-3]
- learning rate_nn, in range [1e-4, 1e-3]

```python
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
    sa = SA(dbn.dbn_get_score, 200, 10, 0.9, var_size, net="DBN")  # initial temp=200, final temp=10, alpha=0.9
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
```

# svm_main

svm_main function is to use pso and sa to optimize parameters of svm:

to switch between "PSO" and "SA", user only need to change `option` 

- if option == "PSO" -- run_svm_pso: input are path to data, upper and under boundaries of all variables

  the parameters of pso are set as: svm.get_score, 2, 5, var_size, net="SVM"

- if option == "SA" -- run_svm_sa: input are path to data, upper and under boundaries of all variables

  he parameters of pso are set as: svm.get_score, 200, 10, 0.9, var_size, net="SVM"

variables to be optimized are C, gamma:

- C, in range [15, 100]
- gamma, in range [0.001, 0.01]

~~~python
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
    sa = SA(svm.get_score, 200, 10, 0.9, var_size, net="SVM")
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
~~~



