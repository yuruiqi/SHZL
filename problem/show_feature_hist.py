import pandas as pd
import matplotlib.pyplot as plt


def show_feature_hist(data_path,feature_name):
    print(feature_name)
    data = pd.read_csv(data_path)

    feature_0 = data[data['label']==0]
    feature_1 = data[data['label']==1]

    plt.hist(feature_0[feature_name], bins=20, alpha=0.7)
    plt.hist(feature_1[feature_name], bins=20, alpha=0.7, color='orange')

    if feature_name.startswith('arterial.nii'):
        title = 'corticomedullar' + feature_name[12:]
    elif feature_name.startswith('venous.nii'):
        title = 'nephrographic' + feature_name[10:]
    elif feature_name.startswith('non_contrast.nii'):
        title = 'non-contrast' + feature_name[16:]
    else:
        print('error')

    plt.title(title)

    plt.show()


# data_path = r'F:\SHZL\model\3d\ISUP\arterial\2kind_feature\feature_data.csv'
# data_path = r'F:\SHZL\model\3d\ISUP\arterial\2kind_feature\data\test_numeric_feature.csv'
# data_path = r'F:\SHZL\model\3d\ISUP\arterial\2kind_feature\data\train_numeric_feature.csv'
# feature_name = 'arterial.nii_original_shape_Sphericity'

data_path = r'F:\SHZL\model\3d\ISUP\ISUP_feature.csv'
feature_name_list = ['non_contrast.nii_original_shape_Sphericity', 'non_contrast.nii_original_shape_Maximum2DDiameterSlice',
                     # 'non_contrast.nii_original_shape_MajorAxisLength','non_contrast.nii_original_firstorder_90Percentile',
                     # 'non_contrast.nii_original_shape_SurfaceVolumeRatio', 'non_contrast.nii_original_shape_LeastAxisLength',
                     # 'non_contrast.nii_original_shape_Maximum2DDiameterColumn', 'non_contrast.nii_original_shape_Maximum3DDiameter',
                     # 'non_contrast.nii_original_shape_SurfaceArea', 'non_contrast.nii_original_shape_MinorAxisLength',
                     # 'non_contrast.nii_original_shape_Maximum2DDiameterRow', 'non_contrast.nii_original_firstorder_Energy',
                     # 'arterial.nii_original_shape_Maximum3DDiameter', 'arterial.nii_original_shape_Sphericity',
                     # 'arterial.nii_original_shape_Maximum2DDiameterSlice', 'arterial.nii_wavelet-LHH_gldm_DependenceEntropy',
                     # 'arterial.nii_wavelet-LHH_gldm_DependenceNonUniformityNormalized', 'arterial.nii_wavelet-LLH_glszm_GrayLevelNonUniformity',
                     'venous.nii_original_shape_Maximum2DDiameterSlice'
                     ]
for feature_name in feature_name_list:
    show_feature_hist(data_path, feature_name)
