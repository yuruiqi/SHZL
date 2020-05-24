import pandas as pd
import matplotlib.pyplot as plt
import os


def show_feature_hist(data_path,feature_name, title):
    print(feature_name)
    data = pd.read_csv(data_path)

    feature_0 = data[data['label']==0]
    feature_1 = data[data['label']==1]

    plt.hist(feature_0[feature_name], bins=20, alpha=0.7)
    plt.hist(feature_1[feature_name], bins=20, alpha=0.7, color='orange')

    savename = ''
    if feature_name.startswith('arterial.nii'):
        savename = 'corticomedullar' + feature_name[12:]
    elif feature_name.startswith('venous.nii'):
        savename = 'nephrographic' + feature_name[10:]
    elif feature_name.startswith('non_contrast.nii'):
        savename = 'non-contrast' + feature_name[16:]
    else:
        print('error')

    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Number of Cases')

    save_path = os.path.join(r'E:\SHZL-paper\pic\figure\supplement', savename+'.tif')
    plt.tight_layout()
    plt.savefig(save_path, format='tif', dpi=300)

    plt.show()


# data_path = r'F:\SHZL\model\3d\ISUP\arterial\2kind_feature\feature_data.csv'
# data_path = r'F:\SHZL\model\3d\ISUP\arterial\2kind_feature\data\test_numeric_feature.csv'
# data_path = r'F:\SHZL\model\3d\ISUP\arterial\2kind_feature\data\train_numeric_feature.csv'
# feature_name = 'arterial.nii_original_shape_Sphericity'

data_path = r'F:\SHZL\model\3d\ISUP\ISUP_feature.csv'
feature_name_list = {'non_contrast.nii_original_shape_Sphericity':'Original Shape \nSphericity',
                     'non_contrast.nii_original_shape_Maximum2DDiameterSlice':'Original Shape \nMaximum2DDiameterSlice',
                     'non_contrast.nii_original_shape_MajorAxisLength': 'Original Shape \nMajorAxisLength',
                     'non_contrast.nii_original_firstorder_90Percentile':'Original Firstorder \n90Percentile',
                     'non_contrast.nii_original_shape_SurfaceVolumeRatio': 'Original Shape \nSurfaceVolumeRatio',
                     'non_contrast.nii_original_shape_LeastAxisLength': 'Original Shape \nLeastAxisLength',
                     'non_contrast.nii_original_shape_Maximum2DDiameterColumn': 'Original Shape \nMaximum2DDiameterColumn',
                     'non_contrast.nii_original_shape_Maximum3DDiameter': 'Original Shape \nMaximum3DDiameter',
                     'non_contrast.nii_original_shape_SurfaceArea': 'Original Shape \nSurfaceArea',
                     'non_contrast.nii_original_shape_MinorAxisLength': 'Original Shape \nMinorAxisLength',
                     'non_contrast.nii_original_shape_Maximum2DDiameterRow': 'Original Shape \nMaximum2DDiameterRow',
                     'non_contrast.nii_original_firstorder_Energy': 'Original Firstorder \nEnergy',
                     'arterial.nii_original_shape_Maximum3DDiameter': 'Original Shape \nMaximum3DDiameter',
                     'arterial.nii_original_shape_Sphericity': 'Original Shape \nSphericity',
                     'arterial.nii_original_shape_Maximum2DDiameterSlice': 'Original Shape \nMaximum2DDiameterSlice',
                     'arterial.nii_wavelet-LHH_gldm_DependenceEntropy': 'Wavelet-LHH GLDM \nDependenceEntropy',
                     'arterial.nii_wavelet-LHH_gldm_DependenceNonUniformityNormalized': 'Wavelet-LHH GLDM \nDependenceNonUniformityNormalized',
                     'arterial.nii_wavelet-LLH_glszm_GrayLevelNonUniformity': 'Wavelet-LLH GLSZM \nGrayLevelNonUniformity',
                     'venous.nii_original_shape_Maximum2DDiameterSlice': 'Original Shape \nMaximum2DDiameterSlice'
                     }
for feature_name in feature_name_list.keys():
    show_feature_hist(data_path, feature_name, feature_name_list[feature_name])

