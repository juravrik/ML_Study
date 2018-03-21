import numpy as np


class LogReg:
    """
    最急降下法によるロジスティック回帰
    """

    def __init__(self, alpha, itr):
        """
        :param alpha: 正則化項の重み
        :param itr: 勾配法の繰り返し回数
        """
        self.alpha = alpha
        self.itr = itr

    def sigmoid(self, x):
        """
        sigmoid関数
        """
        return 1/(1+np.exp(-np.dot(x, self.theta)))

    def logloss(self, x, y):
        """
        logistiloss関数
        """
        return -y*np.log(self.sigmoid(x)) - (1-y)*np.log(1-self.sigmoid(x))

    def fit(self, x, y):
        """
        勾配法によってパラメータを決定
        """
        self.theta = np.random.rand(len(x.T), 1)
        pred_cost = self.logloss(x, y).sum()
        for i in range(self.itr):
            tmp = self.sigmoid(x) - y
            self.theta = self.theta - self.alpha * np.dot(x.T, tmp)

            cur_cost = self.logloss(x, y).sum()
            if (cur_cost > pred_cost).bool():
                break
            pred_cost = cur_cost

    def predict_proba(self, x):
        """
        予測値の確率出力
        """
        return self.sigmoid(x)
