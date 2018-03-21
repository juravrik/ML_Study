import numpy as np


class Fourier():
    """
    基底関数にフーリエ級数を用いた正則化最小二乗法
    """

    def __init__(self, m, t=2*np.pi, lam=0):
        """
        param
        -----
        * m:フィッティングする多項式の次数
        * t:フーリエ級数の周期
        * lam:正則化項の重み
        """
        self.m = m
        self.t = t
        self.lam = lam

    def _CalcDesign(self, x):
        """
        計画行列の導出
        """
        phi = 1/2 * np.ones([len(x), 1])
        for i in range(1, self.m+1):
            sin = np.sin(2.0*np.pi*i*x/self.t)
            cos = np.cos(2.0*np.pi*i*x/self.t)
            phi = np.concatenate((phi, sin, cos), axis=1)
        return phi

    def fit(self, x, y):
        """
        正規方程式を用いて正則化最小二乗法を解く
        """
        design = self._CalcDesign(x)
        n = self.lam * np.identity(2*self.m+1) + np.dot(design.T, design)
        self.theta = np.dot(np.dot(np.linalg.inv(n), design.T), y)

    def pred(self, x):
        """
        予測値の出力
        """
        design = self._CalcDesign(x)
        return np.dot(design, self.theta)
