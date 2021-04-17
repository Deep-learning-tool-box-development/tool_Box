from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from dataset import import_data
from utils import data_FFT
import pickle


def run_knn(path_to_data):

    """
    Main function KNN.

    :param path_to_data: string, Folder of the data files
    :return: None
    """
    x_train, x_test, y_train, y_test = import_data(path_to_data, model='KNN')
    x_train = data_FFT(x_train)  # fft on original dataset
    x_test = data_FFT(x_test)  # fft on original dataset
    knn = KNN(x_train, y_train, x_test, y_test)
    k_range = range(1, 25)  # user input (under, upper bounds)
    weight_choices = ["uniform", "distance"]  # user input, string in list
    params = [k_range, weight_choices]  # editable params
    knn.predict(params)  # user can decide to plot/save result or not
    plot = True    # save: bool, choose to save or not
    save = True    # plot: bool, choose to plot or not
    load = True    # load: bool, choose to load or not
    if plot:
        knn.plot_curve()
    if save:
        knn.save_result(path_to_data)


class KNN:

    def __init__(self, x_train, y_train, x_test, y_test, outdir=None):
        """
        :param x_train: training data
        :param y_train: training label
        :param x_test: test data
        :param y_test: test label
        :param outdir: dir of output
        """
        # print('Selected Network : KNN')
        self.x_train = x_train
        self.y_train = y_train.ravel()
        self.x_test = x_test
        self.y_test = y_test.ravel()
        self.result_error = 0  # initialisation
        self.result_k = 0  # initialisation
        self.k_error = []
        self.k_err_train = []
        self.outdir = outdir
        self.k_range = 1
        self.path = path
        self.knn = None
        self.filename = 'knn_model.sav'

    def predict(self, params):
        """

        :param params: params[0] is number of neighbors;
                    params[1] is weight used in prediction,
                        - "uniform" all points have the same weight,
                        - "distance" weights by the inverse of distance,
                        - [callable] : a user-defined function which accepts an array of distances;
        :return: Error, k value
        """
        # get params
        k_range, weight_choices = params
        self.k_range = k_range
        # Feature Scaling
        sc = StandardScaler()
        self.x_train = sc.fit_transform(self.x_train)
        self.x_test = sc.transform(self.x_test)

        # Fitting K-NN to the Training set
        for weights in weight_choices:
            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric='minkowski', p=2)
                # params[2] and params[3] Default is "minkowski" with p=2, equivalent to the standard Euclidean metric
                # e.g. [5, "uniform",'minkowski', 2]
                # only the first and second params are set to be changed by user
                self.knn = knn
                knn.fit(self.x_train, self.y_train)  # train on train_file
                err = 1 - knn.score(self.x_train, self.y_train)
                Error = 1 - knn.score(self.x_test, self.y_test)  # error on test_file
                print("k=", k)
                print("weights=", weights)
                print("Error", Error)
                self.k_error.append(Error)
                self.k_err_train.append(err)
            self.result_error = min(self.k_error)
            self.result_k = self.k_error.index(self.result_error)+1
            print("best result", "k=", self.result_k, "lowest error=", self.result_error)

        return self.result_error, self.result_k

    def plot_curve(self):
        """
        plot Error_list on variable k value

        :return: None
        """
        plt.plot(self.k_range, self.k_error)
        plt.title("Error under different choice of K")
        plt.xlabel("Value of K for KNN")
        plt.ylabel("Error")
        plt.show()

    def save_result(self, path):
        """
        save result of knn into path

        :param path: string, path of dataset
        """
        pickle.dump(self.knn, open(self.filename, 'wb'))

    def load_model(self):
        loaded_model = pickle.load(open(self.filename, 'rb'))
        result = loaded_model.score(self.x_test, self.y_test)


if __name__ == "__main__":
    """
    Main function to call the knn
    """
    path = './dataset/'
    run_knn(path)

