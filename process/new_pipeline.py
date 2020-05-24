import pandas as pd
import numpy as np
import os
from MeDIT.Statistics import BinaryClassification


# 拆分数据到period
def split_data_to_period(data_list, dir):
    for data_name in data_list:
        data = pd.read_csv(os.path.join(dir, data_name))

        non_contrast_data_path = os.path.join(dir + '/non_contrast', data_name)
        arterial_data_path = os.path.join(dir + '/arterial', data_name)
        venous_data_path = os.path.join(dir + '/venous', data_name)

        non_contrast_index = ['Unnamed: 0', 'label']
        arterial_index = ['Unnamed: 0', 'label']
        venous_index = ['Unnamed: 0', 'label']
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


# 拆分数据到group
def split_data_to_group(data_list,dir):
    for data_name in data_list:
        feature_data = pd.read_csv(os.path.join(dir, data_name))

        # if feature_data.columns.tolist()[0] == 'Unnamed: 0':
        #     feature_data.rename(columns={'Unnamed: 0': 'CaseName'}, inplace=True)

        Casename_data = feature_data['Unnamed: 0']
        label_data = feature_data['label']

        shape_firstorder_feature_data = pd.DataFrame({'CaseName': Casename_data, 'label': label_data})
        gl_feature_data = pd.DataFrame({'CaseName': Casename_data, 'label': label_data})
        trans_feature_data = pd.DataFrame({'CaseName': Casename_data, 'label': label_data})

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

        shape_firstorder_feature_data.to_csv(
            os.path.join(dir, '2kind_feature/' + data_name), index=None)
        data_3kind.to_csv(os.path.join(dir, '3kind_feature/' + data_name), index=None)
        data_4kind.to_csv(os.path.join(dir, '4kind_feature/' + data_name), index=None)


# 拆分fae生成的数据集，到各个period，各个group（需预先创建文件夹）
def split_data(data_list, dir):
    split_data_to_period(data_list, dir)
    for period in ['non_contrast', 'arterial', 'venous']:
        period_dir = os.path.join(dir, period)
        split_data_to_group(data_list, period_dir)


# 根据prob检验分类效果
def examine_model(prob_csv):
    prob_data = pd.read_csv(prob_csv)
    label = prob_data['label'].values.tolist()
    model_names = prob_data.columns.tolist()
    model_names.remove('CaseName')
    model_names.remove('label')

    for model_name in model_names:
        model_prob = prob_data[model_name].values.tolist()
        CS = BinaryClassification(is_show=False)
        CS.Run(model_prob, label)
        print('{}: \tacc: {:.3f},\tsen: {:.3f},\tspe: {:.3f}\n, auc:{:.3f}, {:.3f}-{:.3f}'.
              format(model_name, CS._metric['accuracy'], CS._metric['sensitivity'],
                     CS._metric['specificity'], CS._metric['AUC'],
                     CS._metric['95 CIs Lower'], CS._metric['95 CIs Upper']))


if __name__ == '__main__':
    # 拆分上采样数据集
    # split_data(['test_numeric_feature.csv', 'train_numeric_feature.csv', 'upsampling_features.csv'],
    #            r'F:\SHZL\model\3d\ISUP\upsampling')

    # 拆分小肿瘤数据集(因为代码中是没有命名CaseName，因此删掉名字（Unnamed: 0))
    # split_data(['train_data.csv', 'test_data.csv', 'smote_train_data.csv'],
    #            r'F:\SHZL\model\3d\ISUP\small_tumor')

    # 是用约登指数后，训练集和测试集的模型表现
    examine_model(r'F:\SHZL\model\3d\ISUP\united_model_7\lr\train_prob.csv')
    # examine_model(r'F:\SHZL\model\3d\ISUP\united_model_7\lr\prob.csv')
