import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)


def extract_data(filename):
    category_values = {}
    category_numbers= {}
    category_type   = {}
    names_in_order  = []
    lines = open('data/data_values.txt', 'r').readlines()
    for line in lines:
        name   = line.split(':')[0]
        if name == '| instance weight':
            continue
        classes = [s.strip() for s in line.split(':')[1].split(',')]
        # values
        classes[-1] = classes[-1][:-1]
        category_values[name] = classes
        # num
        category_numbers[name] = len(classes)
        # category_type
        category_type[name] = 'continuous' if classes[0] == 'continuous' else 'nominal'
        # names in order
        names_in_order.append(name)

    class_mapping = {}
    for name in names_in_order:
        if category_type == 'continuous':
            class_mapping[name] = 0
        else:
            class_mapping[name] = {}
            for i in range(category_numbers[name]):
                class_mapping[name][category_values[name][i]] = i


    # 24 is instance weights, we want to ignore it
    cols = [i for i in range(24)] + [i for i in range(25,len(names_in_order) + 1)]
    print('Starting Reading ' + filename)
    df = pd.read_csv(filename, names=names_in_order, usecols=cols, skipinitialspace=True)
    print('Finished Reading ' + filename)

    # cleanse last column
    print('Cleansing income column')
    remove_period = lambda x: str(x).replace('.','')
    df['income'] = df['income'].apply(remove_period)
    # get total number of hot bits
    column_vals = list(df.columns.values)
    total_classes = sum([category_numbers[name] for name in column_vals]) - category_numbers['sex']

    names_in_order.remove('sex')
    # 12 is sex
    print('Extracting labels')
    labels = df['sex'].values
    # data
    raw = df.ix[:, df.columns != 'sex'].values
    m = raw.shape[0]
    r = raw.shape[1]
    k = total_classes

    # labels
    y = np.zeros((m,2))
    for i in range(labels.shape[0]):
        if labels[i] == 'Female':
            y[i,0] = 1
        else:
            y[i,1] = 1

    ## TIME TO FORMAT
    print('Formatting data')
    X = np.zeros((m, k))
    index = 0
    for c in range(r):
        print('Data label number: ' + str(c) + ' of ' + str(r))
        col = raw[:,c]
        name = names_in_order[c]
        if category_type[name] == 'continuous': # normalize it
            col = col.astype(np.float32, copy=False)
            mu = np.mean(col)
            col -= mu
            data = col / np.linalg.norm(col)
            X[:,index] = data
            index += 1
        else:  # encode to one-hot standard
            num_cat = category_numbers[name]
            O = np.zeros((m, num_cat)) - 1
            for cat_row in range(0, m):
                val = str(col[cat_row]).strip()
                if val != '?':
                    try:
                        pos = class_mapping[name][val]
                        O[cat_row,pos] = 1
                    except KeyError:
                        continue
            for copy_col in range(0, num_cat):
                X[:,index] = O[:,copy_col]
                index += 1
    return X, labels

# TEST
train_filename = 'data/census-income.data'
X_train, y_train = extract_data(train_filename)
print('Saving Training Data')
np.save('data/training_data.npy', X_train)
np.save('data/training_labels.npy', y_train)

# TRAIN
test_filename = 'data/census-income.test'
X_test, y_test = extract_data(test_filename)
print('Saving Test Data')
np.save('data/testing_data.npy', X_test)
np.save('data/testing_labels.npy', y_test)

