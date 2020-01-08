import os

from pre_process.RadiomicsFeatureExtractor import RadiomicsFeatureExtractor

# extract features from direction
def extract_feature(dir_path, period_list, out_path_dir):
    """
    dir_path = r'F:\SHZL\data\2d\ICC\group1'
    out_path_dir = r'D:\PycharmProjects\learning\SHZL\feature_data\2d\ICC\group1'
    period_list = ['non_contrast', 'arterial', 'venous']
    """

    for period in period_list:
        print(period)
        out_path_period = os.path.join(out_path_dir, period)
        out_path = os.path.join(out_path_period, period + '.csv')

        extractor = RadiomicsFeatureExtractor(r'H:\SHZL\exampleCT.yaml',
                                              has_label=False)
        extractor.Execute(dir_path,
                          key_name_list=[period+'.nii'],
                          roi_key=[period+'_label'],
                          store_path=out_path)


# dir_path = r'F:\SHZL\data\2d\ICC_new\group3'
# out_path_dir = r'D:\PycharmProjects\learning\SHZL\feature_data\2d\new_ICC\group3'
# period_list = ['non_contrast', 'arterial', 'venous']
#
# extract_feature(dir_path, period_list, out_path_dir)

dir_path = r'F:\SHZL\data\3d\wrong'
out_path_dir = r'F:\SHZL\data\3d\feature'
period_list = ['non_contrast', 'arterial', 'venous']

extract_feature(dir_path, period_list, out_path_dir)
