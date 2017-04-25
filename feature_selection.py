import pandas as pd
import numpy as np

def get_metadata(filename, pruned_columns=[]):
    category_values = {}
    category_numbers= {}
    category_type   = {}
    names_in_order  = []
    lines = open('data/data_values.txt', 'r').readlines()
    for line in lines:
        name = line.split(':')[0]
        # if name == '| instance weight' or name in pruned_columns:
        #     continue
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

    return category_values, category_numbers, category_type, names_in_order, class_mapping

def create_dataframe(filename, className, names_in_order,pruned_columns=[]):
    # create data frame and remove instance weight column
    # cols = range(len(names_in_order))
    # cols = [i for i in range(24)] + [i for i in range(25,len(names_in_order) + 1)]
    print('Starting Reading ' + filename)
    df = pd.read_csv(filename, names=names_in_order, skipinitialspace=True)
    print('Finished Reading ' + filename)

    # cleanse last column
    print('Cleansing income column')
    remove_period = lambda x: str(x).replace('.','')
    df['income'] = df['income'].apply(remove_period)

    # extract labels
    print('Extracting labels')
    labels = df[className].values

    # prune confounding columns
    for prune in pruned_columns:
        if prune in df.columns.values:
            print('dropped ', prune)
            df = df.drop(prune, 1)

    return df,labels

def form_np_array(filename, className, pruned_columns=[]):
    # metadata
    category_values, category_numbers, category_type, names_in_order, class_mapping = get_metadata(filename, pruned_columns)
    # create dataframe
    df, labels = create_dataframe(filename, className, names_in_order, pruned_columns)
    # update names in order
    names_in_order.remove(className)
    for prune in pruned_columns:
        if prune in names_in_order:
            names_in_order.remove(prune)
        if prune in category_values:
            del category_values[prune]
            del category_numbers[prune]
            del category_type[prune]
            del class_mapping[prune]
    # prior counts
    column_vals = list(df.columns.values)
    total_classes = sum([category_numbers[name] for name in column_vals]) - category_numbers[className]
    # data
    raw = df.ix[:, df.columns != className].values
    # dimensions
    m = raw.shape[0] # instances
    r = raw.shape[1] # categorical features
    k = total_classes # feature count after one hot encoding

    # labels
    Y = np.zeros(m)
    for i in range(labels.shape[0]):
        if labels[i] == 'Female':
            Y[i] = 0
        else:
            Y[i] = 1


    X = np.zeros((m, r))
    for col in range(r):
        name = names_in_order[col]
        col_data = raw[:,col]
        if category_type[name] == 'continuous':
            col_data = col_data.astype(np.float32, copy=False)
            mu = np.mean(col_data)
            col_data -= mu
            data = col_data / np.linalg.norm(col_data)
            X[:,col] = data
        else:
            for row in range(m):
                try:
                    if raw[row,col] == '?' or raw[row,col] == 'nan':
                        X[row,col] = -1
                    else:
                        X[row,col] = class_mapping[name][str(raw[row,col]).strip()]
                except KeyError:
                    continue
    return X, Y

if __name__ == '__main__':
    filename = 'data/census-income.data'
    className = 'sex'
    # X_train, y_train = form_np_array('data/census-income.data', 'sex')
    pc = ['year','| instance weight', 'migration prev res in sunbelt', 'migration code-change in msa', \
            'migration code-change in reg', 'migration code-move within reg']
    # pc = []
    category_values, category_numbers, category_type, names_in_order, class_mapping = get_metadata(filename,pruned_columns=pc)
    # df, labels = create_dataframe(filename, className, names_in_order, pruned_columns=pc)
    X,y = form_np_array(filename, className, pruned_columns=pc)

    column_question = np.zeros(X.shape[1])
    total_clean = 0
    for i in range(X.shape[0]):
        clean = True
        for j in range(X.shape[1]):
            if X[i,j] == '?':
                column_question[j] += 1
                clean = False
        if clean:
            total_clean += 1
    m = {}
    for i in range(len(names_in_order) - 1):
        m[names_in_order[i]] = (column_question[i], i, names_in_order[i])
    for key in m:
        print(m[key][2], m[key][1], m[key][0])
    print(total_clean)