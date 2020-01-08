import pandas as pd
import numpy as np
import os

def seperate_1period_features(feature_path,save_dir_path, train_or_test):
    feature_data = pd.read_csv(feature_path)

    if feature_data.columns.tolist()[0] == 'Unnamed: 0':
        feature_data.rename(columns={'Unnamed: 0': 'CaseName'},inplace=True)

    Casename_data = feature_data['CaseName']
    label_data = feature_data['label']

    shape_firstorder_feature_data = pd.DataFrame({'CaseName':Casename_data, 'label':label_data})
    gl_feature_data = pd.DataFrame({'CaseName':Casename_data, 'label':label_data})
    trans_feature_data = pd.DataFrame({'CaseName':Casename_data, 'label':label_data})

    other_feature_num = 0
    for feature_name in feature_data.columns[2:]:
        if ('original' in feature_name) and (('shape' in feature_name) or ('firstorder' in feature_name)):
            shape_firstorder_feature_data[feature_name] = feature_data[feature_name]
        elif ('original' in feature_name) and ('gl' in feature_name):
            gl_feature_data[feature_name] = feature_data[feature_name]
        elif 'wavelet' in feature_name:
            trans_feature_data[feature_name] = feature_data[feature_name]
        else:
            other_feature_num += 1
    print('other_feature_num', other_feature_num)

    data_3kind = pd.merge(shape_firstorder_feature_data, gl_feature_data)
    data_4kind = pd.merge(data_3kind, trans_feature_data)

    shape_firstorder_feature_data.to_csv(os.path.join(save_dir_path, '2kind_feature/'+train_or_test+'_data.csv'), index=None)
    data_3kind.to_csv(os.path.join(save_dir_path, '3kind_feature/'+train_or_test+'_data.csv'), index=None)
    data_4kind.to_csv(os.path.join(save_dir_path, '4kind_feature/'+train_or_test+'_data.csv'), index=None)


def seperate_3period_features(standard_path, train_data_name, test_data_name):
    arterial_save_dir_path = os.path.join(standard_path, 'arterial')
    arterial_train_path = os.path.join(arterial_save_dir_path, train_data_name)
    arterial_test_path = os.path.join(arterial_save_dir_path, test_data_name)

    non_contrast_save_dir_path = os.path.join(standard_path, 'non_contrast')
    non_contrast_train_path = os.path.join(non_contrast_save_dir_path, train_data_name)
    non_contrast_test_path = os.path.join(non_contrast_save_dir_path, test_data_name)

    venous_save_dir_path = os.path.join(standard_path, 'venous')
    venous_train_path = os.path.join(venous_save_dir_path, train_data_name)
    venous_test_path = os.path.join(venous_save_dir_path, test_data_name)

    seperate_1period_features(arterial_train_path, arterial_save_dir_path, 'train')
    seperate_1period_features(arterial_test_path, arterial_save_dir_path, 'test')
    seperate_1period_features(non_contrast_train_path, non_contrast_save_dir_path, 'train')
    seperate_1period_features(non_contrast_test_path, non_contrast_save_dir_path, 'test')
    seperate_1period_features(venous_train_path, venous_save_dir_path, 'train')
    seperate_1period_features(venous_test_path, venous_save_dir_path, 'test')


def seperate_period_data(standard_dir,train_or_test):
    data = pd.read_csv(os.path.join(standard_dir, train_or_test+'_data.csv'))

    non_contrast_data_path = os.path.join(standard_dir+'/non_contrast', train_or_test+'.csv')
    arterial_data_path = os.path.join(standard_dir+'/arterial', train_or_test+'.csv')
    venous_data_path = os.path.join(standard_dir+'/venous', train_or_test+'.csv')

    non_contrast_index = ['CaseName', 'label']
    arterial_index = ['CaseName', 'label']
    venous_index = ['CaseName', 'label']
    for index in data:
        if 'non_contrast' in index:
            non_contrast_index.append(index)
        elif 'arterial' in index:
            arterial_index.append(index)
        elif 'venous' in index:
            venous_index.append(index)

    non_contrast_data = pd.DataFrame(data[non_contrast_index])
    arterial_data = pd.DataFrame(data[arterial_index])
    venous_data = pd.DataFrame(data[venous_index])

    non_contrast_data.to_csv(non_contrast_data_path, index=None)
    arterial_data.to_csv(arterial_data_path, index=None)
    venous_data.to_csv(venous_data_path, index=None)


def seperate_traintest_period_data(standard_dir):
    seperate_period_data(standard_dir, 'train')
    seperate_period_data(standard_dir, 'test')


def seperate_train_test(data_path,save_dir_path,example_path):
    data = pd.read_csv(data_path)
    example_data = pd.read_csv(example_path)

    train_casename = example_data['Unnamed: 0'].tolist()

    train_data = data[data['CaseName'].isin(train_casename)]
    test_data = data[~data['CaseName'].isin(train_casename)]

    train_data_path = os.path.join(save_dir_path, 'train_data.csv')
    test_data_path = os.path.join(save_dir_path, 'test_data.csv')

    train_data.to_csv(train_data_path, index=None)
    test_data.to_csv(test_data_path, index=None)


# standard_dir = r'F:\SHZL\model\3d\ISUP'
# seperate_traintest_period_data(standard_dir)

# standard_path = r'F:\SHZL\model\3d\ISUP'
# train_data_name = 'smote_train.csv'
# test_data_name = 'test.csv'
# seperate_3period_features(standard_path, train_data_name, test_data_name)

seperate_train_test(r'F:\SHZL\model\3d\ISUP\ISUP_feature.csv', r'F:\SHZL\model\3d\ISUP',
                    r'F:\SHZL\model\3d\ISUP\arterial\2kind_feature\train_numeric_feature.csv')
