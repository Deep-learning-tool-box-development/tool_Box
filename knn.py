# import numpy as np
# import os
# from scipy.io import loadmat
# from google.colab import drive
# from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from dataset import import_data
from sklearn.model_selection import train_test_split


def run_knn(path_to_data):

    """
    Main function KNN.
    :param path_to_data: string, Folder of the data files
    :return: None
    """
    x_train, x_test, y_train, y_test = import_data(path_to_data, model='KNN')
    knn = KNN(x_train, y_train, x_test, y_test)
    k_range = range(1, 15)
    weight_choices = ["uniform", "distance"]
    params = [k_range, weight_choices]  # editable params
    knn.predict(params)


class KNN():

    def __init__(self, x_train, y_train, x_test, y_test):
        # print('Selected Network : KNN')
        self.x_train = x_train
        self.y_train = y_train.ravel()
        self.x_test = x_test
        self.y_test = y_test.ravel()
        # self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_train, y_train.ravel(), test_size=0.1)

    # def save_model(self):
    #     from sklearn.externals import joblib
    #     joblib.dump(KNN, 'knn_model.pkl')
    #     # load model
    #     SVM_model = joblib.load('knn_model.pkl')
    #     # make report
    #     from sklearn import metrics
    #     metrics.classification_report(self.y_test, self.result)

    def predict(self, params):
        """
        :param params: params[0] is number of neighbors;
                    params[1] is weight used in prediction,
                        - "uniform" all points have the same weight,
                        - "distance" weights by the inverse of distance,
                        - [callable] : a user-defined function which accepts an array of distances;

        :return: result
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
            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric='minkowski', p=2)
                # params[2] and params[3] Default is "minkowski" with p=2, equivalent to the standard Euclidean metric
                # e.g. [5, "uniform",'minkowski', 2]
                scores = cross_val_score(knn, X_train, self.y_train, scoring='accuracy')
                Error = 1 - scores.mean()
                k_error.append(Error)

                # knn.fit(X_train, self.y_train)
                # Error = 1 - knn.score(X_test, self.y_test)
                # result = knn.predict(X_test)
                # self.result = result
                print("k=", k)
                print("weights=", weights)
                # print("knn_result", result)
                print("Error", Error)
                # k_error.append(Error)
            plt.plot(k_range, k_error)
            plt.title("Error under different choice of K")
            plt.xlabel("Value of K for KNN")
            plt.ylabel("Error")
            plt.show()

        # Predict the Test set results
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.contour3D(k_range, weight_choices, k_error, 50, cmap='binary')
        # ax.set_xlabel('Value of K for KNN')
        # ax.set_ylabel('Weight for KNN')
        # ax.set_zlabel('Error')
        # ax.set_title('3D contour')


if __name__ == "__main__":
    """
    Main function to call the selected model 
    """
    # path = "PATH TO DATASET"
    path = "C:\\Users\\zhudifan\\Desktop\\DL\\dataset\\"
    run_knn(path)
