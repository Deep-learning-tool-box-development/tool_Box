from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


class SVM_Model:
    def __init__(self, x_train, y_train, x_test, y_test, optimization=True):
        """

        :param x_train: training set
        :param y_train: training label
        :param x_test: test set
        :param y_test: test label
        :param optimization: bool
        """
        self.x_train = x_train
        self.y_train = y_train.ravel()  # 将label转为一维数组,shape: (11821, 1)-->(11821,)

        if x_test is None:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_train, self.y_train, random_state=42, test_size=0.2)
        else:
            self.x_test = x_test
            self.y_test = y_test.ravel()
        self.model = None
        self.optimization = optimization

    def get_score(self, params):
        """
        Used as objective function, get model score

        :param params: list, Model hyperparameters
        :return: float, model error
        """
        assert self.optimization is True
        # kernel = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        self.model = SVC(C=params[0], gamma=params[1])
        self.model.fit(self.x_train, self.y_train)  # 训练模型
        # result = self.model.predict(self.x_test) # 对测试集进行分类预测
        Error = 1 - self.model.score(self.x_test, self.y_test)  # 计算测试分类正确率

        return Error

    def train_svm(self, params):
        """
        Usually train SVM

        :param params: list, model params
        :return: None
        """
        assert self.optimization is False
        self.model = SVC(C=params[0], gamma=params[1])
        self.model.fit(self.x_train, self.y_train)  # 训练模型
        # result = self.model.predict(self.x_test) # 对测试集进行分类预测
        acc = self.model.score(self.x_test, self.y_test)
        print("Training complete, with accuracy:", acc)
