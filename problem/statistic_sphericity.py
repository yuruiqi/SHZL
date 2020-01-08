import numpy as np
import pandas as pd
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt


def compute_slice_area(roi_path):
    roi_nii = sitk.ReadImage(roi_path)
    roi_array = sitk.GetArrayFromImage(roi_nii)

    roi_array_trans = np.transpose(roi_array, (1, 2, 0))
    roi_array_trans.astype('float64')

    max_slice_area = 0
    slice_num_total = 0
    for slice_num in range(roi_array_trans.shape[2]):
        slice = roi_array_trans[:, :, slice_num]
        area = np.count_nonzero(slice)

        if area != 0:
            slice_num_total += 1
            if area > max_slice_area:
                max_slice_area = area

    return [slice_num_total, max_slice_area]


def statistic_sphericity(dir_path,save_path):
    case_list = os.listdir(dir_path)

    label_index = ['arterial_label.nii', 'non_contrast_label.nii', 'venous_label.nii']
    area_statistic = [[], [], []]
    slice_statistic = [[], [], []]

    for case in case_list:
        print(case)
        for i in range(3):
            label_path = os.path.join(dir_path+'/'+case, label_index[i])
            slice_statistic[i].append(compute_slice_area(label_path)[0])
            area_statistic[i].append(compute_slice_area(label_path)[1])

    area_df = pd.DataFrame({'CaseName': case_list, 'arterial':area_statistic[0], 'non_contrast':area_statistic[1], 'venous':area_statistic[2]})
    slice_df = pd.DataFrame({'CaseName': case_list, 'arterial':slice_statistic[0], 'non_contrast':slice_statistic[1], 'venous':slice_statistic[2]})

    writer = pd.ExcelWriter(save_path)
    area_df.to_excel(writer, sheet_name='area', index=None)
    slice_df.to_excel(writer, sheet_name='slice', index=None)
    writer.save()
    writer.close()


def add_label(data_path, label_path):
    area_data = pd.read_excel(data_path, sheet_name='area')
    slice_data = pd.read_excel(data_path, sheet_name='slice')
    fuhrman_label_data = pd.read_excel(label_path, sheet_name='Sheet1')
    ISUP_label_data = pd.read_excel(label_path, sheet_name='Sheet2')

    label = []
    for casename in area_data['CaseName']:
        fuhrman_loc = fuhrman_label_data.loc[fuhrman_label_data['Image NO.'] == int(casename[-8:])]
        ISUP_loc = ISUP_label_data.loc[ISUP_label_data['Unnamed: 4'] == int(casename[-8:])]

        fuhrman_label = fuhrman_loc['Unnamed: 8'].values
        ISUP_label = ISUP_loc['Unnamed: 7'].values
        # print(casename, fuhrman_label, ISUP_label)
        if (0 in fuhrman_label) or (0 in ISUP_label):
            label.append(0)
        else:
            label.append(1)
    area_data.insert(1, 'label', label)
    slice_data.insert(1, 'label', label)

    writer = pd.ExcelWriter(data_path)
    area_data.to_excel(writer, sheet_name='area', index=None)
    slice_data.to_excel(writer, sheet_name='slice', index=None)
    writer.save()
    writer.close()


def show_feature_hist(data_path, feature_list=None):
    features_data = pd.read_excel(data_path, sheet_name='slice')
    if feature_list is None:
        feature_list = features_data.columns.values[2:]

    for feature in feature_list:
        ran = (min(features_data[feature]), max(features_data[feature]))
        feature_label_0_distribution = features_data[features_data['label'] == 0][feature]
        feature_label_1_distribution = features_data[features_data['label'] == 1][feature]

        # plt.style.use('ggplot')
        plt.hist(feature_label_0_distribution, edgecolor='black', color='red', alpha=0.7, bins=50, range=ran)
        plt.hist(feature_label_1_distribution, edgecolor='black', color='blue', alpha=0.7, bins=50, range=ran)
        plt.title(feature)
        plt.show()


# dir_path = r'F:\SHZL\data\3d\all'
# save_path = r'F:\SHZL\data\3d\sphericity_problem.xlsx'
# statistic_sphericity(dir_path,save_path)

# data_path = r'F:\SHZL\data\3d\sphericity_problem.xlsx'
# label_path = r'F:\SHZL\ccRCC20190406 分类.xlsx'
# add_label(data_path,label_path)

data_path = r'F:\SHZL\data\3d\sphericity_problem.xlsx'
show_feature_hist(data_path,['non_contrast', 'arterial', 'venous'])
