import numpy as np
from sklearn.ensemble import RandomForestClassifier
from feature_selection import form_np_array
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC

def linear_classifier(X_train, y_train, X_test, y_test):
    # X_train = np.load('data/training_data.npy')
    # y_train = np.load('data/training_labels.npy')
    # X_test = np.load('data/testing_data.npy')
    # y_test = np.load('data/testing_data.npy')

    clf = LinearSVC(verbose=2)
    clf.fit(X_train, y_train)
    print('Linear SVC Training: ', clf.score(X_train, y_train))
    print('Linear SVC Testing: ', clf.score(X_test, y_test))

def build_classifier(X_train, y_train):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    print('Random Forest Training ', clf.score(X_train, y_train))
    return clf

def test_classifier(clf, X_test, y_test):
    print('Random Forest Testing ', clf.score(X_test, y_test))

if __name__ == '__main__':

    pc = ['year','| instance weight', 'migration prev res in sunbelt', 'migration code-change in msa', \
            'migration code-change in reg', 'migration code-move within reg']
    className = 'sex'
    train_filename = 'data/census-income.data'
    test_filename = 'data/census-income.test'
    X_train, y_train = form_np_array(train_filename, className, pruned_columns=pc)
    X_test, y_test = form_np_array(test_filename, className, pruned_columns=pc)

    female_count = (y_train == 0).sum()
    male_count = (y_train == 1).sum()
    total = y_train.shape[0]
    print('Number of Female: ' + str(female_count) + ' ' + str(float(female_count / float(total))))
    print('Number of Males: ' + str(male_count) + ' ' + str(float(male_count / float(total))))

    female_count = (y_test == 0).sum()
    male_count = (y_test == 1).sum()
    total = y_test.shape[0]
    print('Number of Female: ' + str(female_count) + ' ' + str(float(female_count / float(total))))
    print('Number of Males: ' + str(male_count) + ' ' + str(float(male_count / float(total))))
    exit()

    # clf = build_classifier(X_train, y_train)
    # test_classifier(clf, X_test, y_test)
    # feature = clf.feature_importances_
    X_train_full = np.load('data/training_data.npy')
    X_test_full = np.load('data/testing_data.npy')

    # linear_classifier(X_train, y_train, X_test, y_test)
    linear_classifier(X_train_full, y_train, X_test_full, y_test)

