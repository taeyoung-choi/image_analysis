import cv2
import numpy as np
from iris_recognition import IrisLocalization, IrisNormalization, ImageEnhancement, FeatureExtraction, PerformanceEvaluation


def raw_img_process(img, d):
    pupil, outer = IrisLocalization.run(img)
    normalized = IrisNormalization.run(img, pupil, outer)
    enhenced = ImageEnhancement.run(normalized)
    feature = FeatureExtraction.run(enhenced, d)

    return feature


def get_train_feature():
    x_train = []
    y_train = []
    for i in range(1, 109):
        # print(i)
        for j in range(1, 4):
            eye_num = str(i).zfill(3)
            path = '/Users/taeyoungchoi/Documents/Spring 2018/Image Analysis/data/iris/{}/1/{}_1_{}.bmp'.format(
                eye_num, eye_num, j)
            img = cv2.imread(path, 0)
            feature = raw_img_process(img)
            x_train.append(feature)
            y_train.append(i)
    np.savetxt('csv/train_feature.csv', x_train, delimiter=',')
    np.savetxt('csv/train_label.csv', y_train, delimiter=',')

    return np.array(x_train), np.array(y_train)


def get_test_feature():
    x_test = []
    y_test = []
    for i in range(1, 109):
        # print(i)
        for j in range(1, 5):
            eye_num = str(i).zfill(3)
            path = '/Users/taeyoungchoi/Documents/Spring 2018/Image Analysis/data/iris/{}/2/{}_2_{}.bmp'.format(
                eye_num, eye_num, j)
            img = cv2.imread(path, 0)
            feature = raw_img_process(img)
            x_test.append(feature)
            y_test.append(i)
    np.savetxt('csv/test_feature.csv', x_test, delimiter=',')
    np.savetxt('csv/test_label.csv', y_test, delimiter=',')

    return np.array(x_test), np.array(y_test)


def main():
    # x_train, y_train = get_train_feature()
    # x_test, y_test = get_test_feature()
    x_test = np.loadtxt('/Users/taeyoungchoi/Documents/Spring 2018/Image '
                        'Analysis/iris_recognition/csv/test_feature42.csv', delimiter=',')
    y_test = np.loadtxt('/Users/taeyoungchoi/Documents/Spring 2018/Image '
                        'Analysis/iris_recognition/csv/test_label42.csv', delimiter=',')
    x_train = np.loadtxt('/Users/taeyoungchoi/Documents/Spring 2018/Image '
                         'Analysis/iris_recognition/csv/train_feature42.csv', delimiter=',')
    y_train = np.loadtxt('/Users/taeyoungchoi/Documents/Spring 2018/Image '
                         'Analysis/iris_recognition/csv/train_label42.csv', delimiter=',')
    PerformanceEvaluation.rec_rate_by_dimesion(x_train, y_train, x_test, y_test)
    PerformanceEvaluation.fmr(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
