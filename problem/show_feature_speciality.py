import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


# show the distribution of features selected by model
def show_model_features_hist(model_path):
    feature_data_path = os.path.join(model_path, 'selected_feature.csv')

    features_data = pd.read_csv(feature_data_path)
    feature_list = features_data.columns.values[2:]

    for feature in feature_list:
        ran = (min(features_data[feature]),max(features_data[feature]))
        feature_label_0_distribution = features_data[features_data['label'] == 0][feature]
        feature_label_1_distribution = features_data[features_data['label'] == 1][feature]

        # plt.style.use('ggplot')
        plt.hist(feature_label_0_distribution, edgecolor='black', color='red', alpha=0.7, bins=50, range=ran)
        plt.hist(feature_label_1_distribution, edgecolor='black', color='blue', alpha=0.7, bins=50, range=ran)
        plt.title(feature)
        plt.show()


# show distribution of features selected by you
def show_feature_hist(data_path, feature_list=None):
    data = pd.read_csv(data_path)

    features_data = pd.read_csv(data_path)
    if feature_list is None:
        feature_list = features_data.columns.values[2:]

    for feature in feature_list:
        ran = (min(features_data[feature]), max(features_data[feature]))
        feature_label_0_distribution = features_data[features_data['label'] == 0][feature]
        feature_label_1_distribution = features_data[features_data['label'] == 1][feature]

        # plt.style.use('ggplot')
        plt.hist(feature_label_0_distribution, edgecolor='black', color='red', alpha=0.6, bins=50, range=ran)
        plt.hist(feature_label_1_distribution, edgecolor='black', color='green', alpha=0.6, bins=50, range=ran)
        plt.title(feature)
        plt.show()

if __name__ == '__main__':
    # model_path = r'F:\SHZL\model\3d\ISUP\arterial\9.18\model\NormUnit_PCC_ANOVA_2_LR'
    # show_model_features_hist(model_path)

    data_path = r'F:\SHZL\model\3d\ISUP\data_3\test_data.csv'
    show_feature_hist(data_path,['venous.nii_wavelet-HHH_glrlm_ShortRunHighGrayLevelEmphasis'])
