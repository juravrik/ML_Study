import numpy as np


class Kmean:
    def __init__(self, num_type, itr):
        self.num_type = num_type
        self.itr = itr

    def _euclid_distance(self, d1, d2):
        return np.sum(np.sqrt(
    def fit(self, data):
        self.label = np.random.randint(0, self.num_type, data.shape[0])

        for i in range(self.itr):
            for j in range(self.num_type):
                self.centers = data[self.label == j].mean(axis=0)

            for j in data:
                for center in self.centers:




