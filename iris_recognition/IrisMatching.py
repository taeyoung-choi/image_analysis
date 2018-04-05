from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math


def run(x_train, y_train, x_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
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
            val1 = np.sum(np.abs(train_reduced[i, :] - test_reduced[j, :]))
            val2 = np.sum(np.power(train_reduced[i, :] - test_reduced[j, :], 2))
            train = train_reduced[i, :]
            test = test_reduced[j, :]
            train.shape = (1, 107)
            test.shape = (1, 107)
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
