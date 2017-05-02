import numpy as np
from sklearn.ensemble import RandomForestClassifier
from feature_selection import form_np_array
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt

from extract_labels import get_metadata, create_dataframe
from CrossValidate import split_training_data

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

    category_values, category_numbers, category_type, names_in_order, class_mapping = get_metadata(train_filename, pruned_columns=pc)
    df, labels = create_dataframe(train_filename, className, names_in_order, pruned_columns=pc)

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
    # exit()

    # categorical
    # clf = build_classifier(X_train, y_train)
    # names = list(df.columns.values)
    # importances = clf.feature_importances_
    # print('ONE HOT: TRAINING: ' + str(clf.score(X_train, y_train)))
    # print('ONE HOT: TESTING: ' + str(clf.score(X_test, y_test)))

    # print('Feature Ranking: ')
    # a = sorted(zip(map(lambda x: round(x, 4), importances), names),
    #              reverse=True)
    # for elt in a:
    #     print(elt)

    # one hot
    X_train_full = np.load('data/training_data.npy')
    X_test_full = np.load('data/testing_data.npy')
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))

    # estimators = [5, 10, 25, 50, 100]
    # metrics = ['gini', 'entropy']

    # training_acc = np.zeros((len(estimators), len(metrics)))
    # testing_acc = np.zeros(training_acc.shape)

    # K = 5


    # for e_i,n_estimators in enumerate(estimators):
    #     for m_i,metric in enumerate(metrics):
    #         k = 1
    #         for train_X, train_y, test_X, test_y in split_training_data(X_train_full, y_train):
    #             clf = RandomForestClassifier(n_estimators=n_estimators, criterion=metric)
    #             clf.fit(train_X, train_y[:,0])
    #             p_train = clf.predict(train_X)
    #             p_test = clf.predict(test_X)

    #             train = (p_train == train_y[:,0]).sum() / float(train_y.shape[0])
    #             test =  (p_test == test_y[:,0]).sum() / float(test_y.shape[0])
    #             print('Decision Tree Fold ' + str(k) + ' of ' + str(K))
    #             print('\tTrain: ' + str(train) + ' Test: ' + str(test))
    #             training_acc[e_i,m_i] += train
    #             testing_acc[e_i,m_i] += test
    #             k += 1


    # training_acc /= 5
    # testing_acc /= 5

    # h='\\hline'
    # print(h)
    # print('Number of Estimators - Split Criteria & gini & entropy\\\\')
    # print(h)
    # for i in range(len(estimators)):
    #     print(str(estimators[i]) + ' & ' + ' & '.join([str(n) for n in testing_acc[i]]) + '\\\\')
    #     print(h)



    # clf = build_classifier(X_train_full, y_train)
    numE, c = 50, 'gini'
    N = 10
    testing_accuracy = np.zeros(N)
    feat_imp = np.zeros(X_train_full.shape[1])
    for i in range(N):
        print(str(i))
        clf = RandomForestClassifier(n_estimators=numE, criterion=c)
        clf.fit(X_train_full, y_train[:,0])
        test_pred = clf.predict(X_test_full)
        testing_accuracy[i] = (test_pred == y_test[:,0]).sum() / float(y_test.shape[0])
        names = list(df.columns.values)
        feat_imp += clf.feature_importances_
    feat_imp /= 10

    print('\\hline')
    print('Decision Tree: & ' + ' & '.join([str(a) for a in testing_accuracy]))
    print('\\hline')

    # print('ONE HOT: TRAINING: ' + str(clf.score(X_train_full, y_train)))
    # print('ONE HOT: TESTING: ' + str(clf.score(X_test_full, y_test)))

    h = '\\hline'
    print('Feature Ranking: ')
    a = sorted(zip(map(lambda x: round(x, 4), importances), names),
                 reverse=True)
    for elt in a:
        print(h)
        print(elt[1] + ' & ' + elt[0] + '\\\\')
    print(h)



    # linear_classifier(X_train, y_train, X_test, y_test)
    # linear_classifier(X_train_full, y_train, X_test_full, y_test)

