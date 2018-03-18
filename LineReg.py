import pandas as pd
import numpy as np


class Fourier():
    def __init__(self, m, t=2*np.pi, lam=0):
        self.m = m
        self.t = t
        self.lam = lam
        # self.theta = np.random.rand(m)

    def _CalcDesign(self, x):

        phi = pd.DataFrame(1/2 * np.ones(len(x)))
        for i in range(1, self.m+1):
            sin = np.sin(2.0*np.pi*i*x/self.t)
            cos = np.cos(2.0*np.pi*i*x/self.t)
            phi = pd.concat([phi, sin, cos], axis=1)
        return phi

    def fit(self, x, y):
        design = self._CalcDesign(x)
        n = self.lam * np.identity(2*self.m+1) + np.dot(design.T, design)
        self.theta = np.dot(np.dot(np.linalg.inv(n), design.T), y)

    def pred(self, x):
        design = self._CalcDesign(x)
        return np.dot(design, self.theta)
