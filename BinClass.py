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
        return 1/(1+np.exp(-np.dot(x, self.theta)))

    def logloss(self, x, y):
        """
        logistiloss関数
        """
        return -y*np.log(self._sigmoid(x)) - (1-y)*np.log(1-self._sigmoid(x))

    def fit(self, x, y):
        """
        勾配法によってパラメータを決定
        """
        self.theta = np.zeros([len(x.T), 1])
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


class perseptron:

    def __init__(self, eta, itr, epocsize):
        self.eta = eta
        self.itr = itr
        self.epocsize = epocsize

    def fit(self, x, y):
        self.w = np.random.rand(x.shape[1]+1, 1)
        x = np.hstack([np.ones([x.shape[0], 1]), x])
        y = np.where(y == 0, -1, 1)

        def conv(x):
            return np.where(np.dot(x, self.w) > 0, 1, -1)

        for i in range(self.itr):
            epoc = list(np.random.choice(range(x.shape[0]), self.epocsize))
            for tmp_idx in epoc:
                tmp_x = x[tmp_idx, :]
                tmp_y = y[tmp_idx, :]

                grad = (self.eta * tmp_y * tmp_x).reshape(tmp_x.shape[0], 1)
                self.w = np.where(conv(tmp_x) == tmp_y, self.w, self.w + grad)

    def pred(self, x):
        x = np.hstack([np.ones([x.shape[0], 1]), x])
        return np.where(np.dot(x, self.w) > 0, 1, 0)
