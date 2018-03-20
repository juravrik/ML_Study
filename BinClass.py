import numpy as np


class LogReg:

    def __init__(self, alpha, itr):
        self.alpha = alpha
        self.itr = itr

    def sigmoid(self, x):
        return 1/(1+np.exp(-np.dot(x, self.theta)))

    def logloss(self, x, y):
        return -y*np.log(self.sigmoid(x)) - (1-y)*np.log(1-self.sigmoid(x))

    def fit(self, x, y):
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
        return self.sigmoid(x)
