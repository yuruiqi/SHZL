import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# merge all ICC value to ICC.csv
def merge_ICC(dir_path):
    ICC12 = pd.read_csv(os.path.join(dir_path,'ICC12.csv'))
    ICC23 = pd.read_csv(os.path.join(dir_path,'ICC23.csv'))
    ICC13 = pd.read_csv(os.path.join(dir_path,'ICC13.csv'))
    ICC123 = pd.read_csv(os.path.join(dir_path,'ICC123.csv'))

    data_1 = pd.merge(ICC12, ICC23)
    data_2 = pd.merge(data_1, ICC13)
    data_3 = pd.merge(data_2, ICC123)

    data_3.to_csv(os.path.join(dir_path, 'ICC.csv'), index=None)


def del_NaN(data):
    if np.isnan(np.sum(data)) == True:
        to_remain = (1 - np.isnan(data)).astype(np.bool)
        data = data[to_remain]
    return data


# show ICC distribution
def show_ICC_distribution(ICC_path,xmin,ymax):
    ICC_csv = pd.read_csv(ICC_path)

    ICC12 = del_NaN(ICC_csv['ICC12'].values)
    ICC13 = del_NaN(ICC_csv['ICC13'].values)
    ICC23 = del_NaN(ICC_csv['ICC23'].values)
    ICC123 = del_NaN(ICC_csv['ICC123'].values)

    f, ax = plt.subplots(2,2)
    numbins = 50
    range_min = xmin
    ax[0][0].hist(ICC12, numbins, rwidth=0.5, range=(range_min, 1))
    ax[0][1].hist(ICC13, numbins, rwidth=0.5, range=(range_min, 1))
    ax[1][0].hist(ICC23, numbins, rwidth=0.5, range=(range_min, 1))
    ax[1][1].hist(ICC123, numbins, rwidth=0.5, range=(range_min, 1))

    if ymax:
        ax[0][0].set_ylim(0,ymax)
        ax[0][1].set_ylim(0,ymax)
        ax[1][0].set_ylim(0,ymax)
        ax[1][1].set_ylim(0,ymax)

    ax[0][0].set_title('ICC12')
    ax[0][0].set_xlabel('ICC')
    ax[0][0].set_ylabel('Num of Features')

    ax[0][1].set_title('ICC13')
    ax[0][1].set_xlabel('ICC')
    ax[0][1].set_ylabel('Num of Features')

    ax[1][0].set_title('ICC23')
    ax[1][0].set_xlabel('ICC')
    ax[1][0].set_ylabel('Num of Features')

    ax[1][1].set_title('ICC123')
    ax[1][1].set_xlabel('ICC')
    ax[1][1].set_ylabel('Num of Features')

    plt.show()


# compute percentile
def compute_percentile(data_path,percentage):
    data = pd.read_csv(data_path)
    for ICC_name in ['ICC12','ICC13' ,'ICC23', 'ICC123']:
        ICC = data[ICC_name].values
        ICC = del_NaN(ICC)

        percentile = np.percentile(ICC, percentage)
        max = np.max(ICC)
        min = np.min(ICC)
        print(ICC_name,min,max,percentile)


# compute feature numbers which ICC value below threshold
def compute_num_under_threshold(data_path,threshold):
    data = pd.read_csv(data_path)
    for ICC_name in ['ICC12','ICC13' ,'ICC23', 'ICC123']:
        ICC = data[ICC_name].values
        ICC = del_NaN(ICC)
        threshold_destination = np.where(ICC < threshold)
        print(ICC_name,np.shape(threshold_destination)[1])


def compute_corr(ICC_path):
    ICC_csv = pd.read_csv(ICC_path)

    ICC12 = del_NaN(ICC_csv['ICC12'].values)
    ICC13 = del_NaN(ICC_csv['ICC13'].values)
    ICC23 = del_NaN(ICC_csv['ICC23'].values)
    ICC123 = del_NaN(ICC_csv['ICC123'].values)

    print(np.corrcoef(ICC12,ICC13))
    print(np.corrcoef(ICC12,ICC23))
    print(np.corrcoef(ICC13,ICC23))


# select feature by threshold
def select_feature_by_threshold(dimension_dir,ICC_path, threshold):
    all_path = os.path.join(dimension_dir,'all\\all_features.csv')
    save_path = os.path.join(dimension_dir,'all\\selected_features.csv')

    ICC_data = (pd.read_csv(ICC_path)).values
    all_data = pd.read_csv(all_path)

    feature_list = ['CaseName']
    for row in range(np.shape(ICC_data)[0]):
        if (ICC_data[row][2] > threshold) and (ICC_data[row][3] > threshold):
            feature_list.append(ICC_data[row][0])

    selected_data = all_data[feature_list]
    selected_data = delete_redundant_case(selected_data)
    selected_data.to_csv(save_path,index=None)


def select_case_by_standard(model_path, standard_data, feature_data, standard):
    if standard == 'furman':
        standard_to_merge = pd.DataFrame({'CaseName':standard_data['Image NO.'], 'label':standard_data['Unnamed: 8']})
    elif standard == 'ISUP':
        standard_to_merge = pd.DataFrame({'CaseName': standard_data['Unnamed: 4'], 'label': standard_data['Unnamed: 7']})
    else:
        print('standard name wrong')

    standard_output = pd.merge(standard_to_merge,feature_data)
    standard_output = standard_output.drop_duplicates()

    save_path = os.path.join(model_path, standard+"\\"+standard+"_feature.csv")
    standard_output.to_csv(save_path, index=None)

    error_case = set(standard_to_merge['CaseName']).difference(set(feature_data['CaseName']))
    print(standard,error_case)


# i forget
def seperate_furman_isup(all_path, label_path, model_path):
    feature_path = os.path.join(all_path, 'selected_features.csv')
    feature_data = pd.read_csv(feature_path)

    furman_data = pd.read_excel(label_path, sheet_name='Sheet1')
    isup_data = pd.read_excel(label_path, sheet_name='Sheet2')

    for i in range(len(feature_data['CaseName'])):
        feature_data['CaseName'][i] = int(feature_data['CaseName'][i][-8:])

    select_case_by_standard(model_path, furman_data, feature_data, 'furman')
    select_case_by_standard(model_path, isup_data, feature_data, 'ISUP')


def delete_redundant_case(data):
    error_list = ['CHEN_XIAO_CHUN_s  10838430', 'HE_LIN_XUAN_2 10790012', 'CHEN_XIAO_CHUN2_10838430', 'HE_LIN_XUAN2_10790012']
    # for error_name in error_list:
    data = data[~data['CaseName'].isin(error_list)]
    return data


# dir_path = r'D:\PycharmProjects\learning\SHZL\feature_data\2d\new_ICC'
# merge_ICC(dir_path)

# ICC_path = r'D:\PycharmProjects\learning\SHZL\feature_data\2d\new_ICC\ICC.csv'
# show_ICC_distribution(ICC_path, xmin=0.9, ymax=800)
# compute_percentile(ICC_path,10)
# compute_num_under_threshold(ICC_path,0.95)
# compute_corr(ICC_path)

# dimension_path = r'D:\PycharmProjects\learning\SHZL\feature_data\3d'
# ICC_path = r'D:\PycharmProjects\learning\SHZL\feature_data\3d\new_ICC\ICC.csv'
# select_feature_by_threshold(dimension_path,ICC_path,0.9)

label_path = r'F:\SHZL\ccRCC20190406 分类.xlsx'
all_path = r'D:\PycharmProjects\learning\SHZL\feature_data\3d\all'
model_path = r'F:\SHZL\model\3d'
seperate_furman_isup(all_path, label_path, model_path)
