import numpy as np
import pandas as pd
import os
from scipy import stats
import matplotlib.pyplot as plt


# 统计list中各个元素数量
def count_list(input):
    if not isinstance(input, list):
        input = list(input)
    dict = {}
    for i in set(input):
        dict[i] = input.count(i)
    return dict


# 多标签秩和检验
def chi_square_unpaired(array1, array2):
    count1 = count_list(array1)
    count2 = count_list(array2)

    categories = set(list(count1.keys()) + list(count2.keys()))
    contingency_dict = {}
    for category in categories:
        contingency_dict[category] = [count1[category] if count1[category] else 0, count2[category] if count2[category] else 0]
    contingency_pd = pd.DataFrame(contingency_dict)
    contingency_array = contingency_pd.as_matrix()
    return stats.chi2_contingency(contingency_array)


# 检验独立性
def compute_ind(array1, array2, category=False):
    if category:
        print('hist', plt.hist(array1))
        print('hist', plt.hist(array2))
        print(chi_square_unpaired(array1, array2))
    else:
        print(np.mean(array1),'+-', np.std(array1))
        print(np.mean(array2),'+-', np.std(array2))
        all = list(array1) + list(array2)
        print(np.mean(all), '+-', np.std(all))
        _, p = stats.normaltest(all)
        if p > 0.05:
            print(stats.ttest_ind(array1, array2))
        else:
            print(stats.mannwhitneyu(array1, array2))


# 分别得到训练集和测试集的临床特征
def get_set_feature(feature_data, index_name, train_index, test_index):
    train_feature = feature_data[feature_data[index_name].isin(train_index)]
    test_feature = feature_data[feature_data[index_name].isin(test_index)]
    return train_feature, test_feature


# 比较拆分数据集的临床特征独立性
def compare_set():
    train_case = pd.read_csv(r'F:\SHZL\model\3d\ISUP\train_data.csv')['CaseName'].values
    test_case = pd.read_csv(r'F:\SHZL\model\3d\ISUP\test_data.csv')['CaseName'].values
    index = pd.read_excel(r'F:\SHZL\grading.xlsx')
    # 重复项：10632394 10698327 姚五妹10752238low 杨丽群/杨利群10935038high
    feature_data = pd.read_excel(r'F:\SHZL\ccRCC临床特征_去重.xlsx')

    # 根据索引值获得姓名,再获得临床特征
    train_name = index[index['Unnamed: 3'].isin(train_case)]['姓名'].values
    test_name = index[index['Unnamed: 3'].isin(test_case)]['姓名'].values

    train_feature, test_feature = get_set_feature(feature_data, '姓名', train_name, test_name)

    print('ISUP')
    compute_ind(train_feature['ISUP'].values, test_feature['ISUP'].values, category=True)
    print('age')
    compute_ind(train_feature['age'].values, test_feature['age'].values)
    print('size')
    compute_ind(train_feature['size'].values, test_feature['size'].values)
    print('stage')
    compute_ind(train_feature['stage'].values, test_feature['stage'].values, category=True)

    print('gender')
    dict = {'男':0, '女':1}
    gender1 = [dict[x] for x in train_feature['性别'].values]
    gender2 = [dict[x] for x in test_feature['性别'].values]
    compute_ind(gender1, gender2, category=True)


# 因为可能有个data没有输入，所以要try
def try_compare(data1, data2, sentence):
    try:
        _, p = stats.normaltest(data1 + data2)
        if p>0.05:
            _, pvalue = stats.ttest_rel(data1, data2)
            normal = 'normal'
        else:
            _, pvalue = stats.wilcoxon(data1, data2)
            normal = 'not normal'
        if pvalue < 0.05:
            print(sentence, normal, pvalue, '\n')
    except:
        pass


def try_compare_three(data1, data2, data3, sentence):
    try:
        _, p = stats.normaltest(data1 + data2 + data3)
        if p > 0.05:
            _, pvalue = stats.f_oneway(data1, data2, data3)
            normal = 'normal'
        else:
            _, pvalue = stats.friedmanchisquare(data1, data2, data3)
            normal = 'not normal'
        if pvalue < 0.05:
            print(sentence, normal, pvalue, '\n')
    except:
        pass


# 比较三模态形状特征是否有显著差异
def compare_shape():
    non_contrast_data = pd.read_csv(r'F:\SHZL\model\3d\ISUP\non_contrast\selected_non_contrast.csv')
    arterial_data = pd.read_csv(r'F:\SHZL\model\3d\ISUP\arterial\selected_arterial.csv')
    venous_data = pd.read_csv(r'F:\SHZL\model\3d\ISUP\venous\selected_venous.csv')

    # 得到size大小对应特征数据
    # 重复项：10632394 10698327 姚五妹10752238low 杨丽群/杨利群10935038high
    clinical_feature_data = pd.read_excel(r'F:\SHZL\ccRCC临床特征_去重.xlsx')
    # clinical_feature_data = pd.read_excel(r'F:\SHZL\ccRCC临床特征.xlsx')
    index = pd.read_excel(r'F:\SHZL\grading.xlsx')
    small_name = clinical_feature_data[clinical_feature_data['size'] < 4]['姓名'].values
    large_name = clinical_feature_data[clinical_feature_data['size'] >= 4]['姓名'].values
    large_case = index[index['姓名'].isin(large_name)]['Unnamed: 3'].values
    small_case = index[index['姓名'].isin(small_name)]['Unnamed: 3'].values

    non_contrast_large_shape, non_contrast_small_shape = get_set_feature(non_contrast_data, 'CaseName', large_case, small_case)
    arterial_large_shape, arterial_small_shape = get_set_feature(arterial_data, 'CaseName', large_case, small_case)
    venous_large_shape, venous_small_shape = get_set_feature(venous_data, 'CaseName', large_case, small_case)

    # 得到形状特征去掉phase.nii_后的原始名字
    non_contrast_shape_name = [x[17:] for x in non_contrast_data.keys() if 'shape' in x]
    arterial_shape_name = [x[13:] for x in arterial_data.keys() if 'shape' in x]
    venous_shape_name = [x[11:] for x in venous_data.keys() if 'shape' in x]
    feature_set = set(non_contrast_shape_name+arterial_shape_name+venous_shape_name)

    non_contrast_shape = non_contrast_data[['non_contrast.nii_'+x for x in non_contrast_shape_name]]
    arterial_shape = arterial_data[['arterial.nii_'+x for x in arterial_shape_name]]
    venous_shape = venous_data[['venous.nii_'+x for x in venous_shape_name]]

    for feature in feature_set:
        non_contrast_1feature, arterial_1feature, venous_1feature = None, None, None
        non_contrast_large_1feature, arterial_large_1feature, venous_large_1feature = None, None, None
        non_contrast_small_1feature, arterial_small_1feature, venous_small_1feature = None, None, None

        if feature in non_contrast_shape_name:
            non_contrast_1feature = non_contrast_shape['non_contrast.nii_'+feature]
            non_contrast_large_1feature = non_contrast_large_shape['non_contrast.nii_'+feature]
            non_contrast_small_1feature = non_contrast_small_shape['non_contrast.nii_'+feature]
        if feature in arterial_shape_name:
            arterial_1feature = arterial_shape['arterial.nii_'+feature]
            arterial_large_1feature = arterial_large_shape['arterial.nii_'+feature]
            arterial_small_1feature = arterial_small_shape['arterial.nii_'+feature]
        if feature in venous_shape_name:
            venous_1feature = venous_shape['venous.nii_'+feature]
            venous_large_1feature = venous_large_shape['venous.nii_'+feature]
            venous_small_1feature = venous_small_shape['venous.nii_'+feature]

        try_compare(non_contrast_1feature, arterial_1feature, feature + '\nnon_contrast and arterial')
        try_compare(non_contrast_large_1feature, arterial_large_1feature, feature + '\nnon_contrast and arterial: large')
        try_compare(non_contrast_small_1feature, arterial_small_1feature, feature + '\nnon_contrast and arterial: small')

        try_compare(non_contrast_1feature, venous_1feature, feature + '\nnon_contrast and venous')
        try_compare(non_contrast_large_1feature, venous_large_1feature, feature + '\nnon_contrast and venous: large')
        try_compare(non_contrast_small_1feature, venous_small_1feature, feature + '\nnon_contrast and venous: small')

        try_compare(arterial_1feature, venous_1feature, feature + '\narterial and venous')
        try_compare(arterial_large_1feature, venous_large_1feature, feature + '\narterial and venous: large')
        try_compare(arterial_small_1feature, venous_small_1feature, feature + '\narterial and venous: small')

        try_compare_three(non_contrast_1feature, arterial_1feature, venous_1feature, feature + '\nthree: ')
        try_compare_three(non_contrast_large_1feature, arterial_large_1feature, venous_large_1feature, feature + '\nthree: large')
        try_compare_three(non_contrast_small_1feature, arterial_small_1feature, venous_small_1feature, feature + '\nthree: small')


# 比较三期特征，包括被ICC筛去的
def compare_shape_new():
    data = r'D:\PycharmProjects\learning\SHZL\feature_data\3d\all\all_features.csv'
    df = pd.read_csv(data)
    nc_shape_feature = df[[x for x in df.columns if 'non_contrast' in x and 'shape' in x]]
    a_shape_feature = df[[x for x in df.columns if 'arterial' in x and 'shape' in x]]
    v_shape_feature = df[[x for x in df.columns if 'venous' in x and 'shape' in x]]

    for nc_shape_name in nc_shape_feature.columns:
        shape_name = nc_shape_name[17:]
        print(shape_name)
        a_shape_name = 'arterial.nii_'+shape_name
        v_shape_name = 'venous.nii_'+shape_name

        try_compare_three(nc_shape_feature[nc_shape_name], a_shape_feature[a_shape_name], v_shape_feature[v_shape_name],
                          shape_name)


if __name__ == '__main__':
    compare_set()
    # compare_shape()
    # compare_shape_new()
