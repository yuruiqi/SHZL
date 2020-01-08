import numpy as np
import pandas as pd
import os
from scipy import stats
import matplotlib.pyplot as plt


def get_feature_list(feature_name, feature_data, ids):
    id_path = r'F:\SHZL\grading.xlsx'
    id_dic = pd.read_excel(id_path)

    value_list=[]
    for id in ids:
        name = id_dic[id_dic['Unnamed: 3'] == id]['姓名'].values[0]
        value = feature_data[feature_data['姓名'] == name][feature_name].values
        if len(value)==0:
            continue
        value_list.append(value[0])

    return value_list


def ttest(feature_name, train_path, test_path):
    feature_path = r'F:\SHZL\ccRCC临床特征.xlsx'

    feature_data = pd.read_excel(feature_path)
    train_id = pd.read_csv(train_path)['CaseName'].values
    test_id = pd.read_csv(test_path)['CaseName'].values

    train_feature_list = get_feature_list(feature_name, feature_data, train_id)
    test_feature_list = get_feature_list(feature_name, feature_data, test_id)

    # plt.hist(train_feature_list)
    # plt.hist(test_feature_list)
    # plt.title(feature_name)
    # plt.show()

    ttest_result = stats.ttest_ind(train_feature_list, test_feature_list)
    utest_result = stats.mannwhitneyu(train_feature_list, test_feature_list)

    n_train, _, _ = plt.hist(train_feature_list,bins=4)
    n_test, _, _ = plt.hist(test_feature_list,bins=4)
    # 不知道需不要要转置矩阵
    chisquare_array = np.array([n_train,n_test])
    chisquare_result = stats.chi2_contingency(chisquare_array)

    norm_statistic,norm_pvalue = stats.normaltest(feature_data[feature_name].values)

    print(feature_name)
    if norm_pvalue < 0.05:
        print(utest_result)
        if feature_name == 'stage':
            print(chisquare_result)
    else:
        print(ttest_result)

    print('')


ttest('size',r'F:\SHZL\model\3d\ISUP\united_model_7\lr\train_prob.csv',r'F:\SHZL\model\3d\ISUP\united_model_7\lr\prob.csv')
ttest('stage',r'F:\SHZL\model\3d\ISUP\united_model_7\lr\train_prob.csv',r'F:\SHZL\model\3d\ISUP\united_model_7\lr\prob.csv')
ttest('age',r'F:\SHZL\model\3d\ISUP\united_model_7\lr\train_prob.csv',r'F:\SHZL\model\3d\ISUP\united_model_7\lr\prob.csv')
ttest('ISUP',r'F:\SHZL\model\3d\ISUP\united_model_7\lr\train_prob.csv',r'F:\SHZL\model\3d\ISUP\united_model_7\lr\prob.csv')
