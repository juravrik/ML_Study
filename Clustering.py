import numpy as np


class Kmean:
    def __init__(self, num_type, itr):
        self.num_type = num_type
        self.itr = itr

    def _euclid_distance(self, d1, d2):
        return np.sqrt(np.abs(np.sum(d1-d2)))
    
    def fit(self, data):
        self.label = np.random.randint(0, self.num_type, data.shape[0])

        for i in range(self.itr):
            centers = [data[self.label == j].mean(axis=0) for j in range(self.num_type)]
            tmp = self.label.copy()
            for j, d in enumerate(data):
                min_dist=1000000000
                min_label=None
                for k, center in enumerate(centers):
                    dist = self._euclid_distance(d, center) 
                    if dist < min_dist:
                        min_dist = dist
                        min_label = k
                self.label[j] = min_label

            if (tmp == self.label).all():
                return self.label

        return self.label
    
