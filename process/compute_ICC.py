import pandas as pd
import os
import shutil
import numpy as np

# data_try = np.array([[7,8,3,5],
#                      [2,4,4,1],
#                      [1,2,6,1],
#                      [5,5,7,2],
#                      [8,9,5,6],
#                      [9,10,6,7]])

# pick group1 data according to case names in the other group
def pick_group1_data(all_path, group2_path, group1_path):
    pick_list = os.listdir(group2_path)
    for name in pick_list:
        copy_from = os.path.join(all_path,name)
        copy_to = os.path.join(group1_path,name)
        shutil.copytree(copy_from,copy_to)


def compute_BMS(data):
    mean_all = np.mean(data)
    mean_subject = np.mean(data, axis=1)
    n_rater = np.shape(data)[1]
    df = np.shape(data)[0]-1
    SS_between_people = n_rater * sum(np.square(mean_subject-mean_all))
    BMS = SS_between_people/df
    return BMS


def compute_RMS(data):
    mean_all = np.mean(data)
    mean_subject = np.mean(data, axis=0)
    n_rater = np.shape(data)[0]
    df = np.shape(data)[1]-1
    SS_between_items = n_rater * sum(np.square(mean_subject - mean_all))
    RMS = SS_between_items/ df
    return RMS


def compute_EMS(data):
    df_BMS = df = np.shape(data)[0]-1
    df_RMS = df = np.shape(data)[1]-1
    df_EMS = (data.shape[0]-1) * (data.shape[1]-1)
    SS_residual = data.size * np.var(data) - compute_BMS(data)*df_BMS - compute_RMS(data)*df_RMS
    EMS = SS_residual/df_EMS
    return EMS


def compute_ICC2(data):
    BMS = compute_BMS(data)
    RMS = compute_RMS(data)
    EMS = compute_EMS(data)

    n_subject = data.shape[0]
    ICC2 = (BMS-EMS)/(BMS+(RMS-EMS)/n_subject)
    return ICC2


def get_feature_2data(data1_path, data2_path):
    group1_data = pd.read_csv(data1_path)
    group2_data = pd.read_csv(data2_path)
    feature_list = group1_data.columns.values.tolist()[1:]
    ICC_dir = {}
    for feature in feature_list:
        rater1 = np.array(group1_data[feature])
        rater2 = np.array(group2_data[feature])

        feature_data = np.stack((rater1,rater2),axis=1)
        feature_ICC = compute_ICC2(feature_data)
        ICC_dir[feature] = feature_ICC
    return ICC_dir


def get_feature_3data(data1_path, data2_path, data3_path):
    group1_data = pd.read_csv(data1_path)
    group2_data = pd.read_csv(data2_path)
    group3_data = pd.read_csv(data3_path)
    feature_list = group1_data.columns.values.tolist()[1:]
    ICC_dir = {}
    for feature in feature_list:
        rater1 = np.array(group1_data[feature])
        rater2 = np.array(group2_data[feature])
        rater3 = np.array(group3_data[feature])

        feature_data = np.stack((rater1,rater2,rater3),axis=1)
        feature_ICC = compute_ICC2(feature_data)
        ICC_dir[feature] = feature_ICC
    return ICC_dir


def get_feature_4data(data1_path, data2_path, data3_path, data4_path):
    group1_data = pd.read_csv(data1_path)
    group2_data = pd.read_csv(data2_path)
    group3_data = pd.read_csv(data3_path)
    group4_data = pd.read_csv(data4_path)
    feature_list = group1_data.columns.values.tolist()[1:]
    ICC_dir = {}
    for feature in feature_list:
        rater1 = np.array(group1_data[feature])
        rater2 = np.array(group2_data[feature])
        rater3 = np.array(group3_data[feature])
        rater4 = np.array(group4_data[feature])
        print(rater1)
        print(rater2)
        print(rater3)
        print(rater4)

        feature_data = np.stack((rater1,rater2,rater3,rater4),axis=1)
        feature_ICC = compute_ICC2(feature_data)
        ICC_dir[feature] = feature_ICC
    return ICC_dir


def save_ICC_to_csv(ICC_dir, save_csv_path, name):
    data = pd.DataFrame(ICC_dir,index=[name])
    data_trans = pd.DataFrame(data.values.T, index=data.columns, columns=data.index)
    data_trans.to_csv(save_csv_path,columns=None)


def compute_all_ICC(ICC_path):
    group1_path = os.path.join(ICC_path,r'group1\all_features.csv')
    group2_path = os.path.join(ICC_path,r'group2\all_features.csv')
    group3_path = os.path.join(ICC_path,r'group3\all_features.csv')
    save_csv_dir = ICC_path

    ICC_dir_12 = get_feature_2data(group1_path, group2_path)
    ICC_dir_13 = get_feature_2data(group1_path, group3_path)
    ICC_dir_23 = get_feature_2data(group2_path, group3_path)
    ICC_dir_123 = get_feature_3data(group1_path, group2_path, group3_path)

    save_ICC_to_csv(ICC_dir_12, os.path.join(save_csv_dir, 'ICC12.csv'), 'ICC12')
    save_ICC_to_csv(ICC_dir_13, os.path.join(save_csv_dir, 'ICC13.csv'), 'ICC13')
    save_ICC_to_csv(ICC_dir_23, os.path.join(save_csv_dir, 'ICC23.csv'), 'ICC23')
    save_ICC_to_csv(ICC_dir_123, os.path.join(save_csv_dir, 'ICC123.csv'), 'ICC123')


def compute_all_ICC_try():
    group1_path = r'D:\PycharmProjects\learning\SHZL\feature_data\try\ICC\group1\all_features.csv'
    group2_path = r'D:\PycharmProjects\learning\SHZL\feature_data\try\ICC\group2\all_features.csv'
    group3_path = r'D:\PycharmProjects\learning\SHZL\feature_data\try\ICC\group3\all_features.csv'
    group4_path = r'D:\PycharmProjects\learning\SHZL\feature_data\try\ICC\group4\all_features.csv'
    save_csv_dir = r'D:\PycharmProjects\learning\SHZL\feature_data\try\ICC'

    ICC_dir_1234 = get_feature_4data(group1_path, group2_path, group3_path, group4_path)

    save_ICC_to_csv(ICC_dir_1234, os.path.join(save_csv_dir, 'ICC1234.csv'), 'ICC1234')


# print(compute_ICC2(data_try))

# feature = 'non_contrast.nii_log-sigma-1-0-mm-3D_firstorder_10Percentile'

# pick_group1_data(r'F:\SHZL\data\2d\all',r'F:\SHZL\data\2d\ICC_new\group2', r'F:\SHZL\data\2d\ICC_new\group1')


compute_all_ICC(r'D:\PycharmProjects\learning\SHZL\feature_data\2d\new_ICC')

# compute_all_ICC_try()


