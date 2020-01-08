import os
import shutil
import pandas as pd


def ProcessSeriesCopy(case_folder_path, dest_case_folder):
    type_name_list = os.listdir(case_folder_path)
    for type_name in type_name_list:
        print(type_name)
        type_name_path = os.path.join(case_folder_path, type_name)

        for case_name in os.listdir(type_name_path):
            print(case_name)
            case_name_path = os.path.join(type_name_path, case_name)
            dest_folder = os.path.join(dest_case_folder, case_name)
            if not os.path.exists(dest_folder):
                os.mkdir(dest_folder)
            shutil.copy(os.path.join(case_name_path, r'data.nii'), os.path.join(dest_folder, 'new_'+type_name+'.nii'))
            shutil.copy(os.path.join(case_name_path, r'label.nii'), os.path.join(dest_folder, 'new_'+type_name+'_label.nii'))


def CheckData(path):
    for root, dirs, files in os.walk(path):
        if (not dirs) and files:
            if 'data.nii' in files:
                if 'label.nii' not in files:
                    print('lack file:', root, 'label.nii')
            else:
                print('lack file:', root, 'data.nii')


def CompareData(path):
    pass


def RenameFiles(case_folder):
    for root, dirs, files in os.walk(case_folder):
        if (not dirs) and files:
            for file_name in files:
                if not file_name.startswith('new'):
                    for char in file_name:
                        if char.isalpha():
                            char = char.lower()
                            if char == 'n':
                                if not file_name.endswith('label.nii'):
                                    os.rename(os.path.join(root, file_name), os.path.join(root, 'new_non_contrast.nii'))
                                else:
                                    os.rename(os.path.join(root, file_name), os.path.join(root, 'new_non_contrast_label.nii'))
                            elif char == 'a':
                                if not file_name.endswith('label.nii'):
                                    os.rename(os.path.join(root, file_name), os.path.join(root, 'new_arterial.nii'))
                                else:
                                    os.rename(os.path.join(root, file_name), os.path.join(root, 'new_arterial_label.nii'))
                            elif char == 'v':
                                if not file_name.endswith('label.nii'):
                                    os.rename(os.path.join(root, file_name), os.path.join(root, 'new_venous.nii'))
                                else:
                                    os.rename(os.path.join(root, file_name), os.path.join(root, 'new_venous_label.nii'))
                            break


def MergeData():
    df_A = pd.read_csv(r'Z:\SongYang\FAE_data\artery.csv')
    df_N = pd.read_csv(r'Z:\SongYang\FAE_data\non_contrast.csv')
    df_V = pd.read_csv(r'Z:\SongYang\FAE_data\venous.csv')

    result_final = pd.merge(df_A, df_N, sort=True)
    result_final = pd.merge(result_final, df_V, sort=True)

    result_final.to_csv(r'Z:\SongYang\FAE_data\feed_FAE.csv', index=False)


def RenameCaseName():
    orginal = pd.read_csv(r'Z:\SongYang\FAE_data\feed_FAE.csv')
    case_name_col = orginal.pop('CaseName')
    # print(case_name_col)
    for index, item in enumerate(case_name_col):
        # print('index', index)
        # print('item', item)
        temp = item[-8:]
        case_name_col[index] = temp
    print(case_name_col)
    orginal.insert(0, 'CaseName', case_name_col)
    orginal.to_csv(r'Z:\SongYang\FAE_data\data.csv', index=False)


# origin_data = pd.read_excel(r'Z:\SongYang\FAE_data\grading.xlsx', usecols="D,F")
# origin_data.rename(columns={origin_data.columns[0]: 'label'}, inplace=True)
# print(origin_data)
# origin_data.to_csv(r'Z:\SongYang\FAE_data\case_label.csv', index=False)
case_label = pd.read_csv(r'Z:\SongYang\FAE_data\case_label.csv')
data = pd.read_csv(r'Z:\SongYang\FAE_data\data.csv')
final = pd.merge(case_label, data, sort=True)
final.to_csv(r'Z:\SongYang\FAE_data\dataWithLabel.csv', index=False)

# MergeData()
# ProcessSeriesCopy(r'Z:\SongYang\第一次', r'Z:\SongYang\第一次_format')
# CheckData(r'Z:\SongYang\第一次')
# RenameFiles(r'Z:\SongYang\第二次-20190625\RCC-ISUP2019.06.25')

