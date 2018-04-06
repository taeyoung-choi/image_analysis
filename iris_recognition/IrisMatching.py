from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math


def run(x_train, y_train, x_test, n=107):
    """
    Fit Sklearn LinearDiscriminantAnalysis on the feature vectors. Then transforms the feature vector into the
    appropriate dimension, then predicts the result using three different similarity metrics.

    :param x_train: training feature vectors
    :param y_train: training labels
    :param x_test: test feature vectors
    :param n: reduced dimension
    :return: matching labels for each similarity metrics
    """

    # n is a parameter, desired reduced dimension
    lda = LinearDiscriminantAnalysis(n_components=n)
    lda.fit(x_train, y_train)
    # reduce feature vectors to n dimension
    train_reduced = lda.transform(x_train)
    test_reduced = lda.transform(x_test)

    y_predicted_1 = []
    y_predicted_2 = []
    y_predicted_3 = []
    for j in range(432):
        opt1 = math.inf
        opt2 = math.inf
        opt3 = math.inf
        for i in range(324):
            # L1 distance
            val1 = np.sum(np.abs(train_reduced[i, :] - test_reduced[j, :]))
            # L2 distance
            val2 = np.sum(np.power(train_reduced[i, :] - test_reduced[j, :], 2))
            # Cosine Similarity
            train = train_reduced[i, :]
            test = test_reduced[j, :]
            val3 = 1 - cosine_similarity(train, test)
            if val1 < opt1:
                opt1 = val1
                inx1 = y_train[i]
            if val2 < opt2:
                opt2 = val2
                inx2 = y_train[i]
            if val3 < opt3:
                opt3 = val3
                inx3 = y_train[i]
        y_predicted_1.append(inx1)
        y_predicted_2.append(inx2)
        y_predicted_3.append(inx3)

    return {'L1': y_predicted_1, 'L2': y_predicted_2, 'cosine':y_predicted_3}
