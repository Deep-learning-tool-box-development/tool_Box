from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from dataset import import_data
from utils import data_FFT


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
    k_range = range(1, 25)
    weight_choices = ["uniform", "distance"]
    params = [k_range, weight_choices]  # editable params
    knn.predict(params)


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
                        - [callable] : a user-defined function which accepts an array of distances;

        :outcome: plotted out result
        """
        # get params
        k_range, weight_choices = params

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(self.x_train)
        X_test = sc.transform(self.x_test)

        # Fitting K-NN to the Training set
        for weights in weight_choices:
            k_error = []
            k_err_train = []
            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric='minkowski', p=2)
                # params[2] and params[3] Default is "minkowski" with p=2, equivalent to the standard Euclidean metric
                # e.g. [5, "uniform",'minkowski', 2]

                knn.fit(X_train, self.y_train)  # train on train_file
                err = 1 - knn.score(X_train, self.y_train)
                Error = 1 - knn.score(X_test, self.y_test)  # error on test_file

                print("k=", k)
                print("weights=", weights)
                print("Error", Error)
                k_error.append(Error)
                k_err_train.append(err)

            # plot Error_list on variable k value
            self.result_error = min(k_error)
            self.result_k = k_error.index(self.result_error)+1
            print("best result", "k=", self.result_k, "lowest error=", self.result_error)
            plt.plot(k_range, k_error)
            plt.title("Error under different choice of K")
            plt.xlabel("Value of K for KNN")
            plt.ylabel("Error")
            plt.show()


if __name__ == "__main__":
    """
    Main function to call the knn
    """
    # path = "PATH TO DATASET"
    path = './dataset/'
    run_knn(path)
