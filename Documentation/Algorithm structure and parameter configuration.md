basic classes include PSO, SA, CNN , DBN, SVM, KNN interface definition and utils package

# Optimization Algorithms

## PSO class

PSO as optimization algorithms，including 5 functions

1. init function to set up target network, number of the particles, number of iteration, variable size and discrete variable if there is.
2. init_population initialize all the particles within the variable sizes with random numbers, find the global best
3. iterator runs the iteration of the algorithm, at last print the best parameters
4. plot function can plot the optimization curve
5. run function to get pso to run，optimize the target network，print out cost, position and iteration number

```python
class PSO:
    def __init__(self, objective, part_num, num_itr, var_size, candidate=None, net=None):
        """
        Particle Swarm Optimization
        :param objective: cost function as an objective
        :param part_num: integer, number of particles
        :param num_itr: integer, number of iterations
        :param var_size: list, upper and lower bounds of each parameter,
                        as in [[x1_min,x1_max], [x2_min,x2_max],..., [xn_min,xn_max]]
        """
        self.part_num = part_num  # Number of the particles
        self.dim = len(var_size)  # Dimension of the particle
        self.num_itr = num_itr  # Run how many iterations
        self.objective = objective  # Objective function to be optimize
        self.w = 0.9  # initial weight
        self.c1 = 1.49
        self.c2 = 1.49
        self.var_size = var_size  # Length must correspond to the dimension of particle
        self.vmax = 1  # Maximum search velocity
        self.vmin = 0.01  # Minimum search velocity
        self.GlobalBest_Cost = 1e5
        self.GlobalBest_Pos = []
        # Array to hold Best costs on each iterations
        self.Best_Cost = []
        # Save space for particles
        self.particle = []
        assert self.dim == len(self.var_size)
        self.net = net
        self.candidate = candidate

    def init_population(self):
        """
        Initialize all the particles and find the temporary best parameter.
        :return: None
        """

    def iterator(self):
        """
        Run the iterations to find the best parameters.
        :return: None
        """
	def plot_curve(self):
        """
        Plot optimizer curve
        :return: None
        """
        
    def run(self):
        """
        General call for the whole optimization.
        :return: None
        """

```

## SA class

SA as optimization algorithms，including 4 functions

1. init function to set up target network, initial temperature, end temperature, alpha and lower, upper bounds of the parameters in target network
2. random_start function to randomly choose a initial state from the interval
3. random_neighbor to return a random neighbor of current state
4. run function to get sa to run，optimize the target network，print out cost, state, temperature, and iteration number

```python
class SA:
	def __init__(self, objective, initial_temp, final_temp, alpha, var_size, candidate=None, 		net="DBN"):
        """
        :param objective: cost function as an objective
        :param initial_temp: double, manually set initial_temp, e.g. 200
        :param final_temp: double, stop_temp, e.g. 0.1
        :param alpha: double, temperature changing step, [0.900, 0.999]
        :param var_size: list, upper and lower bounds of each parameter
        :param net: choose between "DBN", "CNN", "SVM"
        """
        self.interval = (0, 1)  # set a range (0,1)
        self.objective = objective  # Objective network to be optimize
        self.initial_temp = initial_temp  # 200
        self.final_temp = final_temp  # 10
        self.alpha = alpha  # 0.9 衰减因子 alpha
        self.var_size = var_size  # [[],[],[]]
        self.dim = np.zeros(len(var_size))
        self.net = net
        self.candidate = candidate
        self.temp = []
        self.states = []
        self.costs = []
        self.current_temp = self.initial_temp  # initialisation of temp
    
	def _random_start(self):
   	""" Random start point in the given interval """

	def _random_neighbour(self, state_old):
    """Find neighbour of current state"""
    
	def plot_curve(self):
    """Plot optimizer curve with iteration"""

	def run(self):
    """outcome: cost of all iteration and objective functions' optimized parameters"""

```

# Deep learning algorithms

## CNN class

CNN as DL network, including 1 static function and  6 function inside the class

1. build_network function build CNN network from given parameters
2. init get the training data/label and test data/label
3. train function trains the built network with given parameters
4. cnn_get_score returns the score of the network trained with given outside parameters
5. test function evaluates the model
6. save_model saves the model in the given path
7. report function generates the accuracy report with different metrics

```python
def build_network(dropout=0.5, learning_rate=0.004, num_conv=6):
    """
    Build the CNN network.

    :param num_conv:
    :param learning_rate:
    :param dropout:
    :return: CNN model built from given parameters
    """
class CNN:
	def __init__(self, x_train, y_train,
                 x_test, y_test,
                 discrete_candidate=None,
                 outdir=None,
                 logdir=None,
                 optimization=True,
                 epoch=5):
    """
        :param x_train: training data
        :param y_train: training label
        :param x_test: test data
        :param y_test: test label
        :param outdir: output directory
        :param logdir: log directory
        :param optimization: Bool, is it for optimization
        :param epoch: int, training epoch number
    """
	def train(self, params, plot=False):
    """
        Build and train the CNN, use the given parameters.

        :param params: list, [dropout, learning_rate, batch_size]
        :param plot: bool, plot the learning curve
        :return: training history and the model for evaluate
    """

	def cnn_get_score(self, params):
        """
        Function to get the score of each model.

        :param params: list, [dropout, learning_rate, batch_size]
        :return: float, 1- mean value from last 3 validation accuracy
        """

	def test(self):
        """
        Test trained model
        """
        
	def save_model(self):
        """
        Save cnn model public Function
        """
	def report(self, data, labels):
        """
        Generating network report

        :param data: training data
        :param labels: training label
        :return: None
        """

```

## DBN class

DBN as DL network, including 7 function inside the class

	1. init: Pass in the data set and set some parameters
	2. pretrain: Pre-training, construct n-layer RBM network
	3. finetune: Use the BP algorithm to fine-tune the trained rbm network, modify the weights and bias values
	4. save_model
	5. load_rbm
	6. dbn_get_score: Used to optimize the algorithm. Use the parameters obtained by the optimization algorithm for training

```python
class DBN:
    def __init__(self,
                 train_data,
                 targets,
                 rbm_layers,
                 batch_size,
                 outdir=None,
                 logdir=None,
                 optimization=True):
        """
        Pass in the data set and set some parameters
        :param train_data,targets:training data/label 
        :param rbm_layers:list E.g [400,200,50];number of rbm layers and number of nodes on each layer 
        :param batch_size: int
        :param outdir: Specify the output directory
        :param logdir: Specify the cache directory
        :param optimization: bool Ture:Use optimization algorithms；False：Directly call the DBN network
        """

    def pretrain(self, params, verbose):  
        """
        Pre-training, construct n-layer RBM network
        :param params:params[1] leringrate_rbm
        :param verbose:  0 or 1. Verbosity mode. 0 = silent, 1 = progress bar
        """
        

    def finetune(self, params, verbose): 
        """
        Use the BP algorithm to fine-tune the trained rbm network, modify the weights and bias values
        :param params: params[0]：Dropout；params[2]：lerningrata of finetune 
        :param verbose
        """
       

    def save_model(self):
        model = self.model
        model.save(self.outdir + "dbn_model")


    def load_rbm(self):
        try:
            self.rbm_weights = pickle.load(self.rbm_dir + "rbm_weights.p")
            self.rbm_biases = pickle.load(self.rbm_dir + "rbm_biases.p")
            self.rbm_h_act = pickle.load(self.rbm_dir + "rbm_hidden.p")
        except:
            print("No such file or directory.")

    def dbn_get_score(self, params):
        """
        Use the parameters obtained by the optimization algorithm for training
        :param params: list [Dropout，leringrate_rbm,lerningrata_nn]
        :return: Error rate of validation set
        """
```

# Machine learning algorithms

## SVM

SVM network includes 3 function

 	1. init: Pass in the data set and set some parameters
 	2. get_score: use optimization algorithms to train the best parameters
 	3. train_svm: manually set up parameters to test

~~~python
class SVM_Model():
    def __init__(self, x_train, y_train, x_test, y_test, optimization=True):
        """
        Pass in the data set and set some parameters
        :param x_train, y_train:training daten,Convert label to a one-dimensional array E.g. shape of label: (11821, 1)-->(11821,)
        :param x_test, y_test:test daten. If the test set is not passed in, a part of the training set will be used as the test set 
        :param optimization: bool Ture:Use optimization algorithms；False：Directly call the SVM network
        """

    def get_score(self, params):
        """
        Use the parameters obtained by the optimization algorithm for training
        :param params: list [C,gamma] C:penalty coefficient gamma:parameter of the RBF kernel
        :return: Error rate of test set
        """

    def train_svm(self, params):
        """
        Specify the parameters to train the SVM model
        :param params: list [C,gamma] C:penalty coefficient gamma:parameter of the RBF kernel
        :return: Error rate of test set
        """
~~~



## KNN

KNN algorithm includes 3 functions:

1. init function, input test and train dataset
2. predict function, input: discrete parameters -- k and weight method; output: plotted out curve to show cost result under each iteration
3. run_knn function, input: datapath, get knn function to run

~~~python
class KNN():

    def __init__(self, x_train, y_train, x_test, y_test):
        """
        :param x_train: training data
        :param y_train: training label
        :param x_test: test data
        :param y_test: test label
        """
        # print('Selected Network : KNN')
        self.x_train = x_train
        self.y_train = y_train.ravel()
        self.x_test = x_test
        self.y_test = y_test.ravel()
        self.result_error = 0  # initialisation
        self.result_k = 0  # initialisation

    def predict(self, params):
        """
        :param params: params[0] is number of neighbors;
                    params[1] is weight used in prediction,
                        - "uniform" all points have the same weight,
                        - "distance" weights by the inverse of distance,
        :outcome: plotted out result
        """
        
def run_knn(path_to_data):

    """
    Main function KNN.
    :param path_to_data: string, Folder of the data files
    :return: None
    """
~~~

# Utils

1. translate_params -- translate the list of parameters to the corresponding hyperparameters for CNN
2. plot_learning_curve -- plot curves from training results
4. to_cat -- change the label to One-hot coding
4. report -- Keras API print out test report 
5. import_data -- import data set from given path
6. print_params -- print the network's params via translating function
7. data_FFT -- use fourier transformation to change dataset from time domain into frequency domain

```python
def translate_params(params, candidate):
    """
    Translate the list of parameters to the corresponding parameter.

    :param candidate: list, discrete choices of one parameter in network
    :param params: list, e.g.[dropout, learning_rate, batch_size, number of convolution]
    :return: value of dropout(float), learning_rate(float) and batch_size(int)
    """

def plot_learning_curve(history):
    """
    Plot the learning curve.
    
    :param history: Keras API model training history
    :return: None
    """

def to_cat(data, num_classes=None):
    """
    Change the label to One-hot coding.

    :param data: label to change as in [1, 2, 3, 4]
    :param num_classes: total numbers of classes
    :return: Encoded label
    """
    # Each data should be represents the class number of the data

def report(self, data, labels):
    """
    print out test report
    :param data: test data
    :param labels: test label
    """

def import_data(path, model=None):
    """
    Import data from given path.
    :param path: string, Folder of the data files
    :param model: which model do we use
    :return: ndarray, training and test data
    """
    
def print_params(params, candidate, net=None):
    """
    print the network's params via translating function

    :param candidate: list, discrete choices of one parameter in network
    :param net: String, choose between "CNN", "DBN", "SVM"
    :param params: list of network's parameters
    :return: None
    """
    
def data_FFT(data):
    """
    use fourier transformation to change dataset from time domain into frequency domain
    :param data: input original data
    :return: transformed data
    """
    

```

