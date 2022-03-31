import numpy as np
import random as rd
from math import log


from random_Forest.tree import build_tree, predict
import _pickle as pickle


def load_data(file_name):
    '''Import Data
    input:  file_name(string):File name for saving training data
    output: data_train(list):Training data
    '''
    data_train = []
    f = open(file_name)
    for line in f.readlines():
        lines = line.strip().split("\t")
        data_tmp = []
        for x in lines:
            data_tmp.append(float(x))
        data_train.append(data_tmp)
    f.close()
    return data_train


def choose_samples(data, k):
    '''
    input:  data(list):Original data set
            k(int):Select the number of features
    output: data_samples(list):Selected sample
            feature(list):Selected feature index
    '''
    m, n = np.shape(data)  # The number of samples and the number of sample features
    # 1、Select k features index
    feature = []
    for j in range(k):
        feature.append(rd.randint(0, n - 2))  # n-1 Column is label
    # 2、Select m samples index
    index = []
    for i in range(m):
        index.append(rd.randint(0, m - 1))
    # 3、Select k features of m samples from data to form a data set data_samples
    data_samples = []
    for i in range(m):
        data_tmp = []
        for fea in feature:
            data_tmp.append(data[index[i]][fea])
        data_tmp.append(data[index[i]][-1])
        data_samples.append(data_tmp)
    return data_samples, feature


def random_forest_training(data_train, trees_num):
    '''Build a random forest
    input:  data_train(list):Training data
            trees_num(int):Number of classification trees
    output: trees_result(list):The best division of each tree
            trees_feature(list):The selection of original features in each tree
    '''
    trees_result = []  # Construct the best division of each tree
    trees_feature = []
    n = np.shape(data_train)[1]  # Sample dimension
    if n > 2:
        k = int(log(n - 1, 2)) + 1  # Set the number of features
    else:
        k = 1
    # Start to build every tree
    for i in range(trees_num):
        # 1、Randomly select m samples, k features
        data_samples, feature = choose_samples(data_train, k)
        # 2、Build every classification tree
        tree = build_tree(data_samples)
        # 3、Save the trained classification tree
        trees_result.append(tree)
        # 4、Save the features used in the classification tree
        trees_feature.append(feature)

    return trees_result, trees_feature


def split_data(data_train, feature):
    '''Select feature
    input:  data_train(list):Training data set
            feature(list):Features to choose
    output: data(list):Selected data set
    '''
    m = np.shape(data_train)[0]
    data = []

    for i in range(m):
        data_x_tmp = []
        for x in feature:
            data_x_tmp.append(data_train[i][x])
        data_x_tmp.append(data_train[i][-1])
        data.append(data_x_tmp)
    return data


def get_predict(trees_result, trees_fiture, data_train):
    m_tree = len(trees_result)
    m = np.shape(data_train)[0]

    result = []
    for i in range(m_tree):
        clf = trees_result[i]
        feature = trees_fiture[i]
        data = split_data(data_train, feature)
        result_i = []
        for i in range(m):
            #There is a py version issue here
            #result_i.append((predict(data[i][0:-1], clf).keys())[0])  #py2.X
            #py3.x  The dictionary is first converted to list, and then the index is extracted
            firstSides=list((predict(data[i][0:-1], clf).keys()))
            firstStr=firstSides[0]
            result_i.append(firstStr)

        result.append(result_i)
    final_predict = np.sum(result, axis=0)
    return final_predict


def cal_correct_rate(data_train, final_predict):
    m = len(final_predict)
    corr = 0.0
    for i in range(m):
        if data_train[i][-1] * final_predict[i] > 0:
            corr += 1
    return corr / m


def save_model(trees_result, trees_feature, result_file, feature_file):
    # 1、Save selected features
    m = len(trees_feature)
    f_fea = open(feature_file, "w")
    for i in range(m):
        fea_tmp = []
        for x in trees_feature[i]:
            fea_tmp.append(str(x))
        f_fea.writelines("\t".join(fea_tmp) + "\n")
    f_fea.close()

    # 2、Save the final random forest model

    with open(result_file, 'wb') as f:
        pickle.dump(trees_result, f)


if __name__ == "__main__":
    # 1、Import Data
    print(    "----------- 1、load data -----------")
    data_train = load_data("data.txt")
    # 2、train random_forest Model
    print(    "----------- 2、random forest training ------------")
    trees_result, trees_feature = random_forest_training(data_train, 50)
    # 3、Get trained accuracy
    print(    "------------ 3、get prediction correct rate ------------")
    result = get_predict(trees_result, trees_feature, data_train)
    corr_rate = cal_correct_rate(data_train, result)
    print(    "\t------correct rate: ", corr_rate)
    # 4、Save the final random forest model
    print(    "------------ 4、save model -------------")
    save_model(trees_result, trees_feature, "result_file", "feature_file")