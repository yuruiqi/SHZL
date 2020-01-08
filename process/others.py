import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from process.united_model import get_model_dir_path_by_period


def feature_statistics(data_path):
    data = pd.read_csv(data_path)
    features = data.columns.values.tolist()[1:]
    wavelet = 0
    first_order = 0
    texture = 0
    shape = 0
    log_sigma = 0

    for feature in features:
        if 'wavelet' in feature: wavelet += 1
        elif 'log-sigma' in feature: log_sigma += 1
        elif 'firstorder' in feature: first_order += 1
        elif 'shape' in feature: shape += 1
        elif ('glcm' in feature) or ( 'glszm' in feature) or ( 'glrlm'in feature) or ( 'gldm'in feature):
            texture += 1
        else: print(feature)

    print(len(features))
    print(first_order, shape, texture, wavelet, log_sigma)


def show_feature_weight(model_path,save_path):
    weight_path = os.path.join(model_path,r'LR_coef.csv')
    weight_data = pd.read_csv(weight_path)
    weight_data.sort_values(by='Coef',inplace=True)
    features = weight_data['Unnamed: 0'].values

    broken_point = features[0].index('nii')
    for i in range(len(features)):
        features[i] = features[i][broken_point+4:]

    weights = weight_data['Coef'].values

    plt.figure(figsize=[10,6])
    plt.bar(x=0,bottom=features,orientation="horizontal",height=1,width=weights,edgecolor='b')
    plt.grid(axis='x',linestyle='-.')
    plt.tight_layout()
    plt.savefig(save_path,format='tif',dpi=1000)
    plt.show()


def show_3period_feature_weight(united_model_path):
    dic={}
    feature_num = []
    for period in ['venous','arterial','non_contrast']:
        model_dir_path = get_model_dir_path_by_period(united_model_path,period)
        weight_path = os.path.join(model_dir_path, r'LR_coef.csv')
        weight_data = pd.read_csv(weight_path)
        weight_data.sort_values(by='Coef', inplace=True)

        features = weight_data['Unnamed: 0'].values
        weights = weight_data['Coef'].values
        feature_num.append(len(features))

        for i in range(len(features)):
            if period == 'non_contrast':
                features[i] = features[i].replace(period+'.nii', 'non_contrast')
            if period == 'arterial':
                features[i] = features[i].replace(period+'.nii', 'corticomedullar')
            elif period == 'venous':
                features[i] = features[i].replace(period + '.nii_', 'nephrographic')

            dic[features[i]] = weights[i]

    feature_all = list(dic.keys())
    weight_all = list(dic.values())

    plt.figure(figsize=[10, 6])
    plt.bar(x=0, bottom=feature_all, orientation="horizontal", height=1, width=weight_all, edgecolor='b')
    plt.grid(axis='x', linestyle='-.')
    plt.hlines(feature_num[0]-0.5,-3.5,2, color="red")
    plt.hlines(feature_num[0]+feature_num[1]-0.5,-3.5,2, color="red")
    plt.show()


# feature_statistics(r'F:\SHZL\model\3d\ISUP\venous\selected_venous.csv')
show_feature_weight(r'F:\SHZL\model\3d\ISUP\united_model_7\lr\non_contrast\Norm0Center_PCC_ANOVA_12_LR',
                    save_path=r'E:\SHZL-paper\pic\NCP_features.tif')
show_feature_weight(r'F:\SHZL\model\3d\ISUP\united_model_7\lr\arterial\Norm0Center_PCC_ANOVA_6_LR',
                    save_path=r'E:\SHZL-paper\pic\CMP_features.tif')
show_feature_weight(r'F:\SHZL\model\3d\ISUP\united_model_7\lr\venous\Norm0Center_PCC_ANOVA_1_LR',
                    save_path=r'E:\SHZL-paper\pic\NP_features.tif')
# show_3period_feature_weight(r'F:\SHZL\model\3d\ISUP\united_model_7\lr')
