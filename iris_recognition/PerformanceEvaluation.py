from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import math
import numpy as np
from matplotlib import pyplot as plt


def rec_rate_by_dimesion(x_train, y_train, x_test, y_test):
    accuracy_1 = []
    accuracy_2 = []
    accuracy_3 = []
    x = []
    for k in range(1, 108, 20):
        lda = LinearDiscriminantAnalysis(n_components=k)
        lda.fit(x_train, y_train)
        x.append(k)
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
                train.shape = (1, k)
                test.shape = (1, k)
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
        accuracy_1.append(np.mean(np.array(y_test) == np.array(y_predicted_1)))
        accuracy_2.append(np.mean(np.array(y_test) == np.array(y_predicted_2)))
        accuracy_3.append(np.mean(np.array(y_test) == np.array(y_predicted_3)))
    #cosine 0.715
    np.savetxt('distance1.csv', accuracy_1, delimiter=',')
    np.savetxt('distance2.csv', accuracy_2, delimiter=',')
    np.savetxt('distance3.csv', accuracy_3, delimiter=',')

    print(accuracy_1, accuracy_2, accuracy_3)
    plt.plot(x, accuracy_1, x, accuracy_2, x, accuracy_3)
    plt.legend(['L1', 'L2', 'Cosine'])
    plt.savefig('temp.png')
    plt.close()


def fmr(x_train, y_train, x_test, y_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    roc = []
    pred_prob = lda.predict_proba(x_test)

    for j in [0.4, 0.5, 0.7, 0.8]:
        prediction = lda.predict(x_test).copy()
        for i in range(x_test.shape[0]):
            prob = np.max(pred_prob[i, ])
            if prob < j:
                prediction[i] = 0
        roc.append(prediction)
    false_non_match = []
    false_match = []
    for thresh in roc:
        false_non_match.append(np.mean(thresh == 0))
        false_match.append(np.mean(y_test[thresh != 0] != thresh[thresh != 0]))

    print(false_match)
    print(false_non_match)
    plt.plot(false_match, false_non_match)
    plt.title("False Match and False Non-match Rates")
    plt.xlabel("False Match Rate")
    plt.ylabel("False Non-match Rate")
    plt.savefig("fmr.png")
    plt.close()