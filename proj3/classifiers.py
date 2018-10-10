# ======================================================================= 
# This file is part of the CS519_Project_2 project.
#
# Author: Omid Jafari - omidjafari.com
# Copyright (c) 2018
#
# For the full copyright and license information, please view the LICENSE
# file that was distributed with this source code.
# =======================================================================

import sys
from time import time
import inspect
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# This class will be used to encapsulate all of the classifiers
class Classifiers(object):
    # Constructor
    def __init__(self, eta=0.1, iters=10, seed=1, gamma=0.2, c=1, n=1, p=1, x_tr=[], y_tr=[], x_ts=[]):
        if eta < 0 or eta > 1.0:
            sys.exit("Eta should be between zero and one!")
        self.eta = eta
        self.iters = iters
        self.seed = seed
        self.gamma = gamma
        self.c = c
        self.n = n
        self.p = p
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_ts = x_ts
        self.__obj = None

    def __fit(self):
        start = int(round(time() * 1000))
        self.__obj.fit(self.x_tr, self.y_tr)
        end = int(round(time() * 1000)) - start
        print(inspect.stack()[1][3].split("_", 1)[1] + f" training time: {end if end > 0 else 0} ms")

    def __predict(self):
        start = int(round(time() * 1000))
        y_pred = self.__obj.predict(self.x_ts)
        end = int(round(time() * 1000)) - start
        print(inspect.stack()[1][3].split("_", 1)[1] + f" prediction time: {end if end > 0 else 0} ms")
        return y_pred

    def run_perceptron(self):
        self.__obj = Perceptron(n_iter=self.iters, eta0=self.eta, random_state=self.seed)
        self.__fit()
        return self.__predict()

    def run_linear_svm(self):
        self.__obj = SVC(kernel="linear")
        self.__fit()
        return self.__predict()

    def run_non_linear_svm(self):
        self.__obj = SVC(kernel="rbf", gamma=self.gamma, random_state=self.seed, C=self.c)
        self.__fit()
        return self.__predict()

    def run_decisiontree(self):
        self.__obj = DecisionTreeClassifier(criterion="gini", random_state=self.seed)
        self.__fit()
        return self.__predict()

    def run_knn(self):
        self.__obj = KNeighborsClassifier(n_neighbors=self.n, metric="minkowski", p=self.p)
        self.__fit()
        return self.__predict()

    def run_logisticreg(self):
        self.__obj = LogisticRegression(random_state=self.seed, C=self.c)
        self.__fit()
        return self.__predict()


