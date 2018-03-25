import numpy as np


class LogReg:

    """
    最急降下法によるロジスティック回帰
    """

    def __init__(self, alpha, itr):
        """
        param
        -----
        * alpha: 学習率
        * itr: 勾配法の繰り返し回数
        """
        self.alpha = alpha
        self.itr = itr

    def _sigmoid(self, x):
        """
        sigmoid関数
        """
        ans = 1/(1+np.exp(-np.dot(x, self.theta)))
        eps = 1e-15

        def func(x):
            return max(eps, min(1-eps, ans.any()))
        return np.array(list(map(func, ans))).reshape(x.shape[0], 1)

    def logloss(self, x, y):
        """
        logistiloss関数
        """
        return -y*np.log(self._sigmoid(x)) - (1-y)*np.log(1-self._sigmoid(x))

    def fit(self, x, y):
        """
        勾配法によってパラメータを決定
        """
        self.theta = np.zeros([x.shape[1], 1])
        pred_cost = self.logloss(x, y).sum()
        for i in range(self.itr):

            tmp = self._sigmoid(x) - y
            self.theta = self.theta - self.alpha * np.dot(x.T, tmp)

            cur_cost = self.logloss(x, y).sum()
            if (cur_cost > pred_cost).bool():
                break

            pred_cost = cur_cost

    def predict_proba(self, x):
        """
        予測値の確率出力
        """
        return self._sigmoid(x)

    def pred(self, x):
        return np.where(self._sigmoid(x) > 0.5, 1, 0)


class perseptron:

    def __init__(self, eta, itr, epoc_size):
        self.eta = eta
        self.itr = itr
        self.epoc_size = epoc_size

    def fit(self, x, y):
        self.w = np.random.rand([x.shape[1]+1, 1])
