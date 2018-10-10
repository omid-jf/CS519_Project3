# ======================================================================= 
# This file is part of the CS519_Project_3 project.
#
# Author: Omid Jafari - omidjafari.com
# Copyright (c) 2018
#
# For the full copyright and license information, please view the LICENSE
# file that was distributed with this source code.
# =======================================================================

import sys
from sklearn import datasets
import pandas as pd
import numpy as np
import classifiers
import preprocess_REALDISP

classifier = sys.argv[1]
data_name = sys.argv[2]

# Checking the classifier name
if classifier.lower() not in ["perceptron", "linear_svm", "non_linear_svm", "decisiontree", "knn", "logisticreg"]:
    sys.exit("Invalid classifier name!")

# Checking the dataset name
if data_name.lower() not in ["digits", "realdisp"]:
    sys.exit("Invalid dataset name!")

# Loading the dataset
if data_name.lower() == "digits":
    digits = datasets.load_digits()
    x = digits.data
    y = digits.target
    ds = pd.DataFrame(np.hstack((x, y[:, None])))

else:  # REALDISP
    ds = preprocess_REALDISP.preprocess_REALDISP()
    # x = realdisp.iloc[:, 0:-1]
    # y = realdisp.iloc[:, -1]

# Splitting the data set in to training and testing sets with a fraction of 75%
train = ds.sample(frac=0.75, random_state=1)
test = ds.drop(train.index)

# Extracting the class (target) values
# We are assuming that the last column of the data set is the class
num_cols = ds.shape[1]
x_tr = train.iloc[:, 0:(num_cols - 1)].values
y_tr = train.iloc[:, (num_cols - 1)].values
x_ts = test.iloc[:, 0:(num_cols - 1)].values
y_ts = test.iloc[:, (num_cols - 1)].values


# Run classification
# Parameters:
#       perceptron          eta, iters, seed
#       linear_svm          -
#       non_linear_svm      seed, gamma, c
#       decisiontree        c
#       knn                 n, p
#       logisticreg         seed, c
classify = classifiers.Classifiers(eta=0.001, iters=20, seed=1, gamma=0.2, c=1, n=1, p=1, x_tr=x_tr, y_tr=y_tr, x_ts=x_ts)

if classifier.lower() == "perceptron":
    y_pred = classify.run_perceptron()
elif classifier.lower() == "linear_svm":
    y_pred = classify.run_linear_svm()
elif classifier.lower() == "non_linear_svm":
    y_pred = classify.run_non_linear_svm()
elif classifier.lower() == "decisiontree":
    y_pred = classify.run_decisiontree()
elif classifier.lower() == "knn":
    y_pred = classify.run_knn()
elif classifier.lower() == "logisticreg":
    y_pred = classify.run_logisticreg()

# Evaluating the prediction (percentage of currectly classified samples)
accuracy = ((y_ts == y_pred).sum() / y_ts.shape[0]) * 100
print(classifier.lower() + " accuracy: " + str(accuracy) + "%")
