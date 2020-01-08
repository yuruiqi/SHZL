import os
import numpy as np
import pandas as pd

import SimpleITK as sitk

from MeDIT.Visualization import Imshow3DArray
from roi_checker.ROIChecker import ROIChecker


def normalize(data):
    mx = data.max()
    mn = data.min()
    data = (data-mn)/(mx-mn)
    return data


# rename original data from direction
def rename(dir_path):
    for root,dirs,files in os.walk(dir_path):
        if not dirs:
            for file_name in files:
                for character in file_name:
                    if character == 'n' or character == 'N':
                        if file_name[-9:] == 'label.nii':
                            os.rename(os.path.join(root,file_name),os.path.join(root,'non_contrast_label.nii'))
                        else:
                            os.rename(os.path.join(root,file_name),os.path.join(root,'non_contrast.nii'))
                        break
                    elif character == 'a' or character == 'A':
                        if file_name[-9:] == 'label.nii':
                            os.rename(os.path.join(root,file_name),os.path.join(root,'arterial_label.nii'))
                        else:
                            os.rename(os.path.join(root,file_name),os.path.join(root,'arterial.nii'))
                        break
                    elif character == 'v' or character == 'V':
                        if file_name[-9:] == 'label.nii':
                            os.rename(os.path.join(root,file_name),os.path.join(root,'venous_label.nii'))
                        else:
                            os.rename(os.path.join(root,file_name),os.path.join(root,'venous.nii'))
                        break


# del 'new' in original data name
def rename_del_new(dir_path):
    for root,dirs,files in os.walk(dir_path):
        if not dirs:
            for file in files:
                os.rename(os.path.join(root,file),os.path.join(root,file[4:]))


# check each in the direction
def check_dir(dir_path):
    """
    check_dir(r'F:\SHZL\data\2d\all')
    """

    checker = ROIChecker()
    all_period = ['non_contrast', 'arterial', 'venous']

    for root, dirs, files in os.walk(dir_path):
        if not dirs:
            for file in files:
                if file[-9:] != 'label.nii':
                    data = os.path.join(root, file)
                    roi = os.path.join(root, file[:-4]+'_label.nii')
                    checker.Check(data, roi)
                    if checker.state:
                        print(root, file, checker.state)


def observe(case_dir_path,period_list,show_roi=True):
    for period in period_list:
        print(period)
        data_path = os.path.join(case_dir_path, period + '.nii')
        roi_path = os.path.join(case_dir_path, period + '_label.nii')

        data_sitk = sitk.ReadImage(data_path)
        roi_sitk = sitk.ReadImage(roi_path)

        data_array = sitk.GetArrayFromImage(data_sitk)
        roi_array = sitk.GetArrayFromImage(roi_sitk)

        data_array_trans = np.transpose(data_array, (1, 2, 0))
        roi_array_trans = np.transpose(roi_array, (1, 2, 0))
        data_array_trans.astype('float64')
        roi_array_trans.astype('float64')

        data = normalize(data_array_trans)
        roi = normalize(roi_array_trans)

        if show_roi:
            Imshow3DArray(data, roi)
        else:
            Imshow3DArray(data)


def check_data(data_path,period):
    checker = ROIChecker()
    image = os.path.join(data_path,period+'.nii')
    roi = os.path.join(data_path,period+'_label.nii')
    checker.Check(image, roi)
    print(checker.state)


# merge 3 period features to all_features,csv
def merge_3_period(dir_path):
    """
    merge_3_period(r'D:\PycharmProjects\learning\SHZL\feature_data\2d\ICC\group1')
    """

    non_contrast_path = os.path.join(dir_path, 'non_contrast\\non_contrast.csv')
    arterial_path = os.path.join(dir_path, 'arterial\\arterial.csv')
    venous_path = os.path.join(dir_path, 'venous\\venous.csv')

    non_contrast_data = pd.read_csv(non_contrast_path)
    arterial_data = pd.read_csv(arterial_path)
    venous_data = pd.read_csv(venous_path)

    data_nc_a = pd.merge(non_contrast_data,arterial_data)
    data_na_a_v = pd.merge(data_nc_a,venous_data)

    save_path = os.path.join(dir_path, 'all_features.csv')
    data_na_a_v.to_csv(save_path, index=None)


def replace_case_feature(true_data_path, wrong_data_path, casename):
    true_data = pd.read_csv(true_data_path)
    wrong_data = pd.read_csv(wrong_data_path)

    true_feature = true_data.loc[true_data['CaseName']==casename]
    wrong_data.loc[wrong_data['CaseName']==casename] = true_feature.values

    wrong_data.to_csv(wrong_data_path,index=None)


# rename(r'F:\SHZL\data\3d\ICC_new\group2')
# check_dir(r'F:\SHZL\data\3d\all')
observe(r'F:\SHZL\data\3d\all\BAO_ZHEN_SHENG_10713516',['non_contrast'],show_roi=False)
# check_data(r'F:\SHZL\data\3d\all\WEN_XIAO_HUI_10710409','arterial')
# merge_3_period(r'D:\PycharmProjects\learning\SHZL\feature_data\3d\all')

# replace_case_feature(r'F:\SHZL\data\3d\feature\arterial\arterial.csv',
#                      r'D:\PycharmProjects\learning\SHZL\feature_data\3d\all\arterial\arterial.csv',
#                      'WEN_XIAO_HUI_10710409')
