import numpy as np
import pandas as pd
import os


def GetSigVal(feature_name,df):
    data = df[feature_name].values
    std = np.std(data)
    mean = np.mean(data)
    # print(mean-3*std, mean+3*std)
    sig_val = [x for x in data if (x<mean-3*std or x>mean+3*std)]
    print(feature_name,len(sig_val))


def GetSigVal_bootstrap(feature_name, df):
    data = df[feature_name]
    sample_list = []

    for i in range(1000):
        # if i % 50 == 0:
        #     print('*', end=' ')
        data_resample = data.sample(replace=True, n=data.shape[0])
        mean_sample = np.mean(data_resample.values)
        sample_list.append(mean_sample)

    sample_list = np.array(sample_list)
    mean, std = np.mean(sample_list), np.std(sample_list)

    sig_val = [x for x in data if (x < mean - 3 * std or x > mean + 3 * std)]
    print(mean, std)
    print(feature_name, sig_val)


def GetFeatureNames(feature_path):
    df = pd.read_csv(feature_path)
    feature_names = df['Unnamed: 0'].values.tolist()
    return feature_names


def CheckModelSigVal(bootstrap):
    df = pd.read_csv(r'F:\SHZL\model\3d\ISUP\ISUP_feature.csv')
    # df = pd.read_csv(r'D:\PycharmProjects\learning\SHZL\feature_data\3d\all\all_features.csv')
    feature_dir_list = [r'F:\SHZL\model\3d\ISUP\united_model_7\lr\arterial\Norm0Center_PCC_ANOVA_6_LR',
                        r'F:\SHZL\model\3d\ISUP\united_model_7\lr\non_contrast\Norm0Center_PCC_ANOVA_12_LR',
                        r'F:\SHZL\model\3d\ISUP\united_model_7\lr\venous\Norm0Center_PCC_ANOVA_1_LR']

    for feature_dir in feature_dir_list:
        feature_path = os.path.join(feature_dir, 'LR_coef.csv')
        for feature_name in GetFeatureNames(feature_path):
            if not bootstrap:
                GetSigVal(feature_name,df)
            elif bootstrap:
                GetSigVal_bootstrap(feature_name, df)


def CheckModelSigVal_other(feature_list,bootstrap):
    df = pd.read_csv(r'F:\SHZL\model\3d\ISUP\model_compare\other_features.csv')
    for feature_name in feature_list:
        if not bootstrap:
            GetSigVal(feature_name, df)
        elif bootstrap:
            GetSigVal_bootstrap(feature_name, df)


other_features = ['arterial.nii_wavelet-HLH_glszm_ZoneEntropy',
                  'arterial.nii_wavelet-HHL_glrlm_LongRunLowGrayLevelEmphasis',
                  'arterial.nii_wavelet-HHL_glszm_LargeAreaHighGrayLevelEmphasis',
                  'arterial.nii_original_glszm_LargeAreaLowGrayLevelEmphasis',
                  # 'venous.nii_exponential_glrlm_RunVariance',
                  'venous.nii_wavelet-HHL_firstorder_RootMeanSquared',
                  'arterial.nii_wavelet-LLH_glcm_SumEntropy']

if __name__ == '__main__':
    # CheckModelSigVal(bootstrap=False)
    CheckModelSigVal_other(bootstrap=False, feature_list=other_features)
