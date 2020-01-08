import numpy as np
import pandas as pd
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score
from imblearn.metrics import specificity_score
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon


def get_model_dir_path_by_period(united_model_path, period):
    period_model_path = os.path.join(united_model_path, period)
    model_dir_path = os.path.join(period_model_path,os.listdir(period_model_path)[0])
    return model_dir_path


def zero_center_normalize(data, normalize_path):
    normalize_way = pd.read_csv(normalize_path)
    feature_list = normalize_way['feature_name'].tolist()

    period_feature_data = data[feature_list]
    mean = normalize_way['interception'].values
    std = normalize_way['slop'].values
    period_normalized_data = (period_feature_data - mean)/std

    return period_normalized_data


def print_result(name,train_label,prob_train,pred_train,test_label,prob_test,pred_test):
    auc_train = roc_auc_score(train_label, prob_train)
    auc_test = roc_auc_score(test_label, prob_test)
    acc_train = accuracy_score(train_label, pred_train)
    acc_test = accuracy_score(test_label, pred_test)
    sen_train = recall_score(train_label, pred_train)
    sen_test = recall_score(test_label, pred_test)
    spe_train = specificity_score(train_label, pred_train)
    spe_test = specificity_score(test_label, pred_test)

    space = ' '*len(name)

    print('{} \n'
          'training: auc:{:.3f},\tsen: {:.3f},\tsen: {:.3f},\tspe: {:.3f}\n'
          'testing : auc:{:.3f},\tsen: {:.3f},\tsen: {:.3f},\tspe: {:.3f}'
          .format(name, auc_train, acc_train,sen_train, spe_train, auc_test,acc_test, sen_test, spe_test))


def predict_result(model_dir_path, data):
    model_path = os.path.join(model_dir_path, 'model.pickle')
    feature_path = os.path.join(model_dir_path, 'selected_feature.csv')
    normalize_path = os.path.join(model_dir_path, 'zero_center_normalization_training.csv')

    # 归一化
    data_norm = zero_center_normalize(data, normalize_path)

    feature_data = pd.read_csv(feature_path)
    feature_list = feature_data.columns.values[2:]
    train_data = data_norm[feature_list].values
    label = data['label'].values
    # print(train_data)

    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        prob = model.predict_proba(train_data)[:, 1]
        pred = model.predict(train_data)
    return prob,pred


def unite_3model(united_model_path, train_path, test_path):
    train_data = pd.read_csv(train_path)
    train_label = train_data['label'].values

    non_contrast_model_dir_path = get_model_dir_path_by_period(united_model_path, 'non_contrast')
    prob_non_contrast,pred_non_contrast = predict_result(non_contrast_model_dir_path, train_data)

    arterial_model_dir_path = get_model_dir_path_by_period(united_model_path, 'arterial')
    prob_arterial,pred_arterial = predict_result(arterial_model_dir_path, train_data)

    venous_model_dir_path = get_model_dir_path_by_period(united_model_path, 'venous')
    prob_venous,pred_venous = predict_result(venous_model_dir_path, train_data)

    # 开始获得三期联合模型
    prob_all = np.stack([prob_non_contrast, prob_arterial, prob_venous], axis=1)
    test_data = pd.read_csv(test_path)
    test_label = test_data['label'].values

    lr_model = LogisticRegression(solver='liblinear')
    lr_model.fit(prob_all, train_label)

    with open(os.path.join(united_model_path,'3period_united_model.pickle'), 'wb') as f:
        pickle.dump(lr_model, f)

    train_pred = lr_model.predict(prob_all)
    train_prob = lr_model.predict_proba(prob_all)[:, 1]

    # 开始预测test！
    prob_non_contrast_test,pred_non_contrast_test = predict_result(non_contrast_model_dir_path, test_data)
    prob_arterial_test,pred_arterial_test = predict_result(arterial_model_dir_path, test_data)
    prob_venous_test,pred_venous_test = predict_result(venous_model_dir_path, test_data)

    print_result('non_contrast',train_label,prob_non_contrast,pred_non_contrast,test_label,prob_non_contrast_test,pred_non_contrast_test)
    print_result('arterial',train_label,prob_arterial,pred_arterial,test_label,prob_arterial_test,pred_arterial_test)
    print_result('venous',train_label,prob_venous,pred_venous,test_label,prob_venous_test,pred_venous_test)

    prob_all_test = np.stack([prob_non_contrast_test, prob_arterial_test, prob_venous_test], axis=1)
    # print(np.concatenate([prob_all,train_label[:,np.newaxis]],axis=1))
    # print(np.concatenate([prob_all_test,test_label[:,np.newaxis]],axis=1))

    test_pred = lr_model.predict(prob_all_test)
    test_prob = lr_model.predict_proba(prob_all_test)[:, 1]

    print_result('unite',train_label,train_prob,train_pred,test_label,test_prob,test_pred)
    print('coef and intercept', lr_model.coef_, lr_model.intercept_)
    print('')


def unite_2model(united_model_path, train_path, test_path , period_list):
    train_data = pd.read_csv(train_path)
    train_label = train_data['label'].values

    model1_dir_path = get_model_dir_path_by_period(united_model_path, period_list[0])
    prob1, pred1 = predict_result(model1_dir_path, train_data)

    model2_dir_path = get_model_dir_path_by_period(united_model_path, period_list[1])
    prob2, pred2 = predict_result(model2_dir_path, train_data)

    # 开始获得三期联合模型
    prob_all = np.stack([prob1, prob2], axis=1)
    test_data = pd.read_csv(test_path)
    test_label = test_data['label'].values

    lr_model = LogisticRegression(solver='liblinear')
    lr_model.fit(prob_all, train_label)

    with open(os.path.join(united_model_path,period_list[0] + '_' + period_list[1] + '_united_model.pickle'), 'wb') as f:
        pickle.dump(lr_model, f)

    train_pred = lr_model.predict(prob_all)
    train_prob = lr_model.predict_proba(prob_all)[:, 1]

    # 开始预测test！
    prob1_test, pred1_test = predict_result(model1_dir_path, test_data)
    prob2_test, pred2_test = predict_result(model2_dir_path, test_data)

    # print_result(period_list[0], train_label, prob1, pred1, test_label, prob1_test,pred1_test)
    # print_result(period_list[1], train_label, prob2, pred2, test_label, prob2_test,pred2_test)

    prob_all_test = np.stack([prob1_test, prob2_test], axis=1)
    # print(np.concatenate([prob_all,train_label[:,np.newaxis]],axis=1))
    # print(np.concatenate([prob_all_test,test_label[:,np.newaxis]],axis=1))

    test_pred = lr_model.predict(prob_all_test)
    test_prob = lr_model.predict_proba(prob_all_test)[:, 1]

    print_result('{} and {}'.format(period_list[0], period_list[1]), train_label, train_prob, train_pred, test_label, test_prob, test_pred)
    print('coef and intercept', lr_model.coef_, lr_model.intercept_)
    print('')


if __name__ == '__main__':
    united_model_path = r'F:\SHZL\model\3d\ISUP\united_model_7\lr'
    # train_path = r'F:\SHZL\model\3d\ISUP\old_data\train_data.csv'
    # test_path = r'F:\SHZL\model\3d\ISUP\old_data\test_data.csv'
    train_path = r'F:\SHZL\model\3d\ISUP\train_data.csv'
    test_path = r'F:\SHZL\model\3d\ISUP\test_data.csv'

    unite_3model(united_model_path, train_path, test_path)
    unite_2model(united_model_path, train_path, test_path, ['non_contrast', 'arterial'])
    unite_2model(united_model_path, train_path, test_path, ['arterial', 'venous'])
    unite_2model(united_model_path, train_path, test_path, ['non_contrast', 'venous'])
