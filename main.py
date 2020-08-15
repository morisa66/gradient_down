import numpy as np


class Model:
    def __init__(self, X, y, epoch=100, lr=1.0e-3):
        """
        :param X: np.mat,source data,
        if you wan to calculate normal equation,it can't be singular matrix
        :param y: np.mat,target data
        :param epoch: num of iteration
        :param lr: learning rate
        :return np.mat,the weights
        """
        self.X = X
        self.y = y
        self.epoch = epoch
        self.lr = 1.0e-5
        self.w = np.mat(np.random.normal(loc=0.0, scale=1.0, size=[self.X.shape[1]])).squeeze().transpose()
        self.normal_equation = np.mat(np.random.normal(loc=0.0, scale=1.0, size=[self.X.shape[1], 1]))

    def calculate_gradient_down(self):
        for i in range(self.epoch):
            self.w -= self.lr * self.X.transpose() * (self.X * self.w - self.y)

    def calculate_normal_equation(self):
        self.normal_equation = (self.X.transpose() * self.X).I * self.X.transpose() * self.y

    def predict(self, X, use_normal_equation=False):
        """
        :param X: np.mat,the data for prediction
        :param use_normal_equation: use normal equation for prediction or not
        :return:prediction
        """
        if not use_normal_equation:
            return X * self.w
        else:
            return X * self.normal_equation


def test():
    X = np.mat(np.random.randint(0, 100, size=[20, 3]), dtype=np.float64)
    w = np.mat([3, 2, 1], dtype=np.float64).squeeze(axis=0).transpose()
    y = X * w

    Xtest = np.mat(np.random.randint(0, 100, size=[20, 3]), dtype=np.float64)
    ytest = Xtest * w

    model = Model(X, y)
    model.calculate_gradient_down()
    model.calculate_normal_equation()
    print("w:", model.w.transpose())
    print("normal_equation:", model.normal_equation.transpose())

    err1 = model.predict(Xtest) - ytest
    err2 = model.predict(Xtest, use_normal_equation=True) - ytest
    print("w predict error:", err1.transpose() * err1)
    print("normal_equation predict error:", err1.transpose() * err2)


if __name__ == '__main__':
    """
    w: [[3.00000005 1.99999983 1.00000014]]
    normal_equation: [[3. 2. 1.]]
    w predict error: [[6.36796808e-10]]
    normal_equation predict error: [[5.57834374e-18]]
    """
    test()
