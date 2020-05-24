import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score
from imblearn.metrics import specificity_score
# import classification_statistics
from MeDIT.Statistics import BinaryClassification
from scipy import stats
from imblearn.over_sampling import SMOTE


# 归一化
def zero_center_normalize(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


# 生成只包含提取出来的(非)形状特征的数据集
def generate_set(csv_path, shape=False, size=False):
    shape_feature = ['non_contrast.nii_original_shape_Sphericity',
                     'non_contrast.nii_original_shape_Maximum2DDiameterSlice',
                     'non_contrast.nii_original_shape_MajorAxisLength',
                     'non_contrast.nii_original_shape_SurfaceVolumeRatio',
                     'non_contrast.nii_original_shape_LeastAxisLength',
                     'non_contrast.nii_original_shape_Maximum2DDiameterColumn',
                     'non_contrast.nii_original_shape_Maximum3DDiameter',
                     'non_contrast.nii_original_shape_SurfaceArea',
                     'non_contrast.nii_original_shape_MinorAxisLength',
                     'non_contrast.nii_original_shape_Maximum2DDiameterRow',
                     'arterial.nii_original_shape_Maximum3DDiameter',
                     'arterial.nii_original_shape_Sphericity',
                     'arterial.nii_original_shape_Maximum2DDiameterSlice',
                     'venous.nii_original_shape_Maximum2DDiameterSlice']
    non_shape_feature = ['non_contrast.nii_original_firstorder_90Percentile',
                         'non_contrast.nii_original_firstorder_Energy',
                         'arterial.nii_wavelet-LLH_glszm_GrayLevelNonUniformity',
                         'arterial.nii_wavelet-LHH_gldm_DependenceNonUniformityNormalized',
                         'arterial.nii_wavelet-LHH_gldm_DependenceEntropy']

    data = pd.read_csv(csv_path)

    if size == 'small':
        small_casename = get_small_tumor_case()
        data = data[data['Unnamed: 0'].isin(small_casename)]

    label = data['label'].values
    shape_data = data[[x for x in data.columns.values if x in shape_feature]].values
    non_shape_data = data[[x for x in data.columns.values if x in non_shape_feature]].values

    if shape == 'shape':
        return shape_data, label
    elif shape == 'non_shape':
        return non_shape_data, label
    else:
        return data, label


# 确定特征的情况下，得到单期态模型
def get_phase_model(train_path, test_path, shape, size):
    train_data, train_label = generate_set(train_path, shape, size)
    test_data, test_label = generate_set(test_path, shape, size)

    train_data = zero_center_normalize(train_data)
    test_data = zero_center_normalize(test_data)

    lr_model = LogisticRegression(solver='liblinear')
    lr_model.fit(train_data, train_label)

    train_pred = lr_model.predict(train_data)
    train_prob = lr_model.predict_proba(train_data)[:, 1]

    test_pred = lr_model.predict(test_data)
    test_prob = lr_model.predict_proba(test_data)[:, 1]

    return train_pred, train_prob, train_label, test_pred, test_prob, test_label


# 根据单期态结果得到联合模型，并把结果存入csv
def get_unite_model(save_path, shape, size):
    non_contrast_train_csv = r'F:\SHZL\model\3d\ISUP\non_contrast\4kind_feature\smote_features.csv'
    arterial_train_csv = r'F:\SHZL\model\3d\ISUP\arterial\4kind_feature\smote_features.csv'
    venous_train_csv = r'F:\SHZL\model\3d\ISUP\venous\4kind_feature\smote_features.csv'

    non_contrast_test_csv = r'F:\SHZL\model\3d\ISUP\non_contrast\4kind_feature\test_numeric_feature.csv'
    arterial_test_csv = r'F:\SHZL\model\3d\ISUP\arterial\4kind_feature\test_numeric_feature.csv'
    venous_test_csv = r'F:\SHZL\model\3d\ISUP\venous\4kind_feature\test_numeric_feature.csv'

    # 不同期态的训练、测试集的prob和pred
    nc_train_pred, nc_train_prob, nc_train_label, nc_test_pred, nc_test_prob, nc_test_label = \
        get_phase_model(non_contrast_train_csv, non_contrast_test_csv, shape, size)
    a_train_pred, a_train_prob, a_train_label, a_test_pred, a_test_prob, a_test_label = \
        get_phase_model(arterial_train_csv, arterial_test_csv, shape, size)
    v_train_pred, v_train_prob, v_train_label, v_test_pred, v_test_prob, v_test_label = \
        get_phase_model(venous_train_csv, venous_test_csv, shape, size)

    # 三期联合的prob，以及label
    all_train_prob = np.stack([nc_train_prob, a_train_prob, v_train_prob], axis=1)
    all_train_prob = np.stack([nc_train_prob, a_train_prob], axis=1)
    all_train_label = nc_train_label
    all_test_prob = np.stack([nc_test_prob, a_test_prob, v_test_prob], axis=1)
    all_test_label = nc_test_label

    lr_model_unite = LogisticRegression(solver='liblinear')
    lr_model_unite.fit(all_train_prob, all_train_label)

    # 得到的联合模型的pred,prob
    test_pred = lr_model_unite.predict(all_test_prob)
    test_prob = lr_model_unite.predict_proba(all_test_prob)[:,1]

    df = pd.DataFrame({'label': all_test_label,
                       'non_contrast_pred': nc_test_pred, 'non_contrast_prob': nc_test_prob,
                       'arterial_pred': a_test_pred, 'arterial_prob': a_test_prob,
                       'venous_pred': nc_test_pred, 'venous_prob': nc_test_prob,
                       'united_pred': test_pred, 'united_prob': test_prob})
    df.to_csv(save_path, index=None)


# 根据csv比较结果
def compare_model():
    all_feature_prob_result = pd.read_csv(r'F:\SHZL\model\3d\ISUP\united_model_7\lr\prob.csv')
    shape_result = pd.read_csv(r'F:\SHZL\model\3d\ISUP\review_reply\shape_model_result.csv')

    all_prob = all_feature_prob_result['n+a+v']

    label = shape_result['label'].tolist()
    label = [round(x) for x in label]
    united_pred = shape_result['united_pred'].tolist()
    united_prob = shape_result['united_prob'].tolist()

    CS = BinaryClassification(is_show=False)
    CS.Run(united_prob, label)
    print(CS._metric)
    # print('auc: ',classification_statistics.get_auc(united_prob, label, draw=False))
    print(stats.wilcoxon(all_prob, united_prob))


# 得到小肿瘤的casename
def get_small_tumor_case():
    index = pd.read_excel(r'F:\SHZL\grading.xlsx')
    feature_data = pd.read_excel(r'F:\SHZL\ccRCC临床特征.xlsx')

    # 根据size得到姓名，在获得casename,再得到feature
    small_tumor_name = feature_data[feature_data['size'] < 4]['姓名'].values
    small_tumor_casename = index[index['姓名'].isin(small_tumor_name)]['Unnamed: 3']

    return small_tumor_casename


# 比较小肿瘤模型结果
def compare_small_tumor():
    small_casename = get_small_tumor_case()

    # 统计肿瘤大小、高低分数量
    train_data = pd.read_csv(r'F:\SHZL\model\3d\ISUP\train_data.csv')
    test_data = pd.read_csv(r'F:\SHZL\model\3d\ISUP\test_data.csv')

    small_train_data = train_data[train_data['CaseName'].isin(small_casename)]
    small_test_data = test_data[test_data['CaseName'].isin(small_casename)]

    n_small_train_high = small_train_data[small_train_data['label'] == 1].shape[0]
    n_small_train_low = small_train_data[small_train_data['label'] == 0].shape[0]
    n_small_test_high = small_test_data[small_test_data['label'] == 1].shape[0]
    n_small_test_low = small_test_data[small_test_data['label'] == 0].shape[0]
    print('small:\ntrain:high %d, low%d\ntest:high %d, low%d'%(n_small_train_high, n_small_train_low, n_small_test_high, n_small_test_low))

    large_train_data = train_data[~train_data['CaseName'].isin(small_casename)]
    large_test_data = test_data[~test_data['CaseName'].isin(small_casename)]

    n_large_train_high = large_train_data[large_train_data['label'] == 1].shape[0]
    n_large_train_low = large_train_data[large_train_data['label'] == 0].shape[0]
    n_large_test_high = large_test_data[large_test_data['label'] == 1].shape[0]
    n_large_test_low = large_test_data[large_test_data['label'] == 0].shape[0]
    print('large:\ntrain:high %d, low%d\ntest:high %d, low%d' % (n_large_train_high, n_large_train_low, n_large_test_high, n_large_test_low))

    # 在原有模型上的小肿瘤表现
    pred_result = pd.read_csv(r'F:\SHZL\model\3d\ISUP\united_model_7\lr\pred.csv')
    prob_result = pd.read_csv(r'F:\SHZL\model\3d\ISUP\united_model_7\lr\prob.csv')

    label = pred_result['label']
    prob = prob_result['n+a+v']
    prob_n_a = prob_result['n+a']

    small_label = pred_result[pred_result['CaseName'].isin(small_casename)]['label']
    small_prob = prob_result[prob_result['CaseName'].isin(small_casename)]['n+a+v']
    small_prob_n_a = prob_result[prob_result['CaseName'].isin(small_casename)]['n+a']

    large_label = pred_result[~pred_result['CaseName'].isin(small_casename)]['label']
    large_prob = prob_result[~prob_result['CaseName'].isin(small_casename)]['n+a+v']
    large_prob_n_a = prob_result[~prob_result['CaseName'].isin(small_casename)]['n+a']

    CS =BinaryClassification(is_show=False)

    CS.Run(prob_n_a.tolist(), label.tolist())
    print('all\n', CS._metric)

    CS.Run(small_prob_n_a.tolist(), small_label.tolist())
    print('small\n', CS._metric)

    CS.Run(large_prob_n_a.tolist(), large_label.tolist())
    print('large\n', CS._metric)

    # 重新训练后的效果（由于没有用smote，作废）
    # new_small_result = pd.read_csv(r'F:\SHZL\model\3d\ISUP\review_reply\small_tumor_result.csv')
    # new_small_pred = new_small_result['united_pred']
    # new_small_prob = new_small_result['united_prob']
    # CS.Run(new_small_prob.tolist(), small_label.tolist())
    # print(CS._metric)



def save_small_tumor_to_csv():
    dir = r'F:\SHZL\model\3d\ISUP\small_tumor'
    train_data = pd.read_csv(r'F:\SHZL\model\3d\ISUP\train_data.csv')
    test_data = pd.read_csv(r'F:\SHZL\model\3d\ISUP\test_data.csv')

    small_casename = get_small_tumor_case().tolist()
    df_cols = ['CaseName', 'label'] + small_casename
    small_train_data = train_data[train_data['CaseName'].isin(df_cols)]
    small_test_data = test_data[test_data['CaseName'].isin(df_cols)]

    small_train_data.to_csv(os.path.join(dir, 'train_data.csv'), index=None)
    small_test_data.to_csv(os.path.join(dir, 'test_data.csv'), index=None)


if __name__ == '__main__':
    # 比较只提取shape特征
    # get_unite_model(r'F:\SHZL\model\3d\ISUP\review_reply\shape_model_result.csv', shape='shape', size=False)
    # compare_model()

    # 比较只选用小肿瘤
    # get_unite_model(r'F:\SHZL\model\3d\ISUP\review_reply\small_tumor_result.csv', shape=False, size='small')
    compare_small_tumor()

    # 根据原来的train test，生成小肿瘤数据集
    # save_small_tumor_to_csv()
