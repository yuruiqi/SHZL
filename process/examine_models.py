import numpy as np
import pandas as pd
import os
import pickle
from scipy.stats import wilcoxon
from process.united_model import predict_result
from process.united_model import get_model_dir_path_by_period
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score
from imblearn.metrics import specificity_score
from scipy import stats
import matplotlib.pyplot as plt


def wilcoxon_test(united_model_path,test_data_path):
    # non_contrast_model_dir_path = get_model_dir_path_by_period(united_model_path, 'non_contrast')
    # arterial_model_dir_path = get_model_dir_path_by_period(united_model_path, 'arterial')
    # venous_model_dir_path = get_model_dir_path_by_period(united_model_path, 'venous')
    #
    # test_data = pd.read_csv(test_data_path)
    # prob_non_contrast,pred_non_contrast = predict_result(non_contrast_model_dir_path, test_data)
    # prob_arterial,pred_arterial = predict_result(arterial_model_dir_path, test_data)
    # prob_venous,pred_venous = predict_result(venous_model_dir_path, test_data)
    #
    # prob_best,pred_best = get_prediction_by_period(united_model_path,['non_contrast','arterial', 'venous'],test_data_path)

    df = pd.read_csv(r'F:\SHZL\model\3d\ISUP\united_model_7\lr\prob.csv')
    len_col = len(df.columns)
    for i in range(2,len_col-1):
        model1 = df.columns[i]
        prob_model1 = df[model1]
        for j in range(i+1, len_col):
            model2 = df.columns[j]
            prob_model2 = df[model2]

            print('{} and {}: {}'.format(model1,model2,wilcoxon(prob_model1-prob_model2)))

    # print('{} and {}: {}'.format('non_contrast','arterial',wilcoxon(prob_non_contrast-prob_arterial)))
    # print('{} and {}: {}'.format('arterial','venous',wilcoxon(prob_arterial-prob_venous)))
    # print('{} and {}: {}'.format('non_contrast','venous',wilcoxon(prob_non_contrast-prob_venous)))
    # print('{} and {}: {}'.format('best','non_contrast',wilcoxon(prob_best-prob_non_contrast)))


def predict_unite(united_model_path, test_data, period_list):
    test_label = test_data['label'].values

    prob_array = []
    for period in period_list:
        model_dir_path = get_model_dir_path_by_period(united_model_path,period)
        prob, pred = predict_result(model_dir_path, test_data)
        prob_array.append(prob)
    prob_array = np.array(prob_array).T

    if len(period_list) == 3:
        unite_path = os.path.join(united_model_path,'3period_united_model.pickle')
    else:
        unite_path = os.path.join(united_model_path,period_list[0] + '_' + period_list[1] + '_united_model.pickle')

    with open(unite_path, 'rb') as file:
        united_model = pickle.load(file)
        prob_unite = united_model.predict_proba(prob_array)[:,1]
        pred_unite = united_model.predict(prob_array)

    return prob_unite,pred_unite


def compute_ic(united_model_path, test_data_path, period_list):
    test_data = pd.read_csv(test_data_path)
    case_num = test_data['label'].shape[0]

    auc_list = []
    for i in range(1000):
        if i % 50 == 0:
            print('*', end=' ')
        test_data_resample = test_data.sample(n=case_num, replace=True)
        label_resample = test_data_resample['label']
        prob, pred = predict_unite(united_model_path, test_data_resample, period_list)
        auc = roc_auc_score(label_resample,prob)
        auc_list.append(auc)

    auc_list = np.array(auc_list)
    mean, std = auc_list.mean(), auc_list.std(ddof=1)
    conf_intveral = stats.norm.interval(0.95, loc=mean, scale=std)
    print(period_list,conf_intveral)
    return auc_list


def compute_period_ic(united_model_path, test_data_path,period):
    test_data = pd.read_csv(test_data_path)
    case_num = test_data['label'].shape[0]
    model_dir_path = get_model_dir_path_by_period(united_model_path, period)

    auc_list = []
    for i in range(1000):
        if i%50 == 0:
            print('*',end=' ')
        test_data_resample = test_data.sample(n=case_num,replace=True)
        label_resample = test_data_resample['label']
        prob, pred = predict_result(model_dir_path, test_data_resample)
        auc = roc_auc_score(label_resample,prob)
        auc_list.append(auc)

    auc_list = np.array(auc_list)
    mean, std = auc_list.mean(), auc_list.std(ddof=1)
    conf_intveral = stats.norm.interval(0.95, loc=mean, scale=std)
    print(period,conf_intveral)
    return auc_list


def get_prediction_by_period(united_model_path,model_period,test_data_path):
    test_data = pd.read_csv(test_data_path)

    if isinstance(model_period,list):
        prob,pred = predict_unite(united_model_path,test_data,model_period)
        return prob,pred
    else:
        model_dir_path = get_model_dir_path_by_period(united_model_path, model_period)
        prob,pred = predict_result(model_dir_path, test_data)
        return prob,pred


def t_test(united_model_path,model1_period,model2_period,test_data_path):
    if isinstance(model1_period,list):
        auc1 = compute_ic(united_model_path,test_data_path,model1_period)
    else:
        auc1 = compute_period_ic(united_model_path,test_data_path,model1_period)

    if isinstance(model2_period,list):
        auc2 = compute_ic(united_model_path,test_data_path,model2_period)
    else:
        auc2 = compute_period_ic(united_model_path,test_data_path,model2_period)

    print(stats.ttest_ind(auc1,auc2))

    plt.hist(auc1, bins=20, alpha=0.7)
    plt.hist(auc2, bins=20, alpha=0.7, color='orange')

    plt.show()


def t_test_all(united_model_path,test_data_path):
    non_contrast_model_dir_path = get_model_dir_path_by_period(united_model_path, 'non_contrast')
    arterial_model_dir_path = get_model_dir_path_by_period(united_model_path, 'arterial')
    venous_model_dir_path = get_model_dir_path_by_period(united_model_path, 'venous')

    test_data = pd.read_csv(test_data_path)

    auc_n = compute_period_ic(united_model_path, test_data_path, 'non_contrast')
    auc_a = compute_period_ic(united_model_path, test_data_path, 'arterial')
    auc_v = compute_period_ic(united_model_path, test_data_path, 'venous')

    auc_n_a = compute_ic(united_model_path, test_data_path, ['non_contrast','arterial'])
    auc_n_v = compute_ic(united_model_path, test_data_path, ['non_contrast','venous'])
    auc_a_v = compute_ic(united_model_path, test_data_path, ['arterial','venous'])
    auc_three = compute_ic(united_model_path, test_data_path, ['non_contrast','arterial','venous'])

    print('{} and {}:{}'.format('non_contrast','arterial',stats.ttest_ind(auc_n, auc_a)))
    print('{} and {}:{}'.format('non_contrast','venous',stats.ttest_ind(auc_n, auc_v)))
    print('{} and {}:{}'.format('non_contrast','n_a',stats.ttest_ind(auc_n, auc_n_a)))
    print('{} and {}:{}'.format('non_contrast','a_v',stats.ttest_ind(auc_n, auc_a_v)))
    print('{} and {}:{}'.format('non_contrast','n_v',stats.ttest_ind(auc_n, auc_n_v)))
    print('{} and {}:{}'.format('non_contrast','united',stats.ttest_ind(auc_n, auc_three)))

    print('{} and {}:{}'.format('arterial','venous',stats.ttest_ind(auc_a, auc_v)))
    print('{} and {}:{}'.format('arterial','n_a',stats.ttest_ind(auc_a, auc_n_a)))
    print('{} and {}:{}'.format('arterial','a_v',stats.ttest_ind(auc_a, auc_a_v)))
    print('{} and {}:{}'.format('arterial','n_v',stats.ttest_ind(auc_a, auc_n_v)))
    print('{} and {}:{}'.format('arterial','united',stats.ttest_ind(auc_a, auc_three)))

    print('{} and {}:{}'.format('venous','n_a',stats.ttest_ind(auc_v, auc_n_a)))
    print('{} and {}:{}'.format('venous','a_v',stats.ttest_ind(auc_v, auc_a_v)))
    print('{} and {}:{}'.format('venous','n_v',stats.ttest_ind(auc_v, auc_n_v)))
    print('{} and {}:{}'.format('venous','united',stats.ttest_ind(auc_v, auc_three)))

    print('{} and {}:{}'.format('n_a','a_v',stats.ttest_ind(auc_n_a, auc_a_v)))
    print('{} and {}:{}'.format('n_a','n_v',stats.ttest_ind(auc_n_a, auc_n_v)))
    print('{} and {}:{}'.format('n_a','united',stats.ttest_ind(auc_n_a, auc_three)))

    print('{} and {}:{}'.format('n_v','a_v',stats.ttest_ind(auc_n_v, auc_a_v)))
    print('{} and {}:{}'.format('n_v','united',stats.ttest_ind(auc_n_v, auc_three)))

    print('{} and {}:{}'.format('a_v','united',stats.ttest_ind(auc_a_v, auc_three)))


def print_result_to_csv(united_model_path,data_path,save_dir,save_name):
    test_data = pd.read_csv(data_path)
    casename = test_data['CaseName']
    label = test_data['label']

    prob_n, pred_n = get_prediction_by_period(united_model_path,'non_contrast',data_path)
    prob_a, pred_a = get_prediction_by_period(united_model_path,'arterial',data_path)
    prob_v, pred_v = get_prediction_by_period(united_model_path,'venous',data_path)

    prob_n_a, pred_n_a = get_prediction_by_period(united_model_path,['non_contrast', 'arterial'],data_path)
    prob_n_v, pred_n_v = get_prediction_by_period(united_model_path,['non_contrast', 'venous'],data_path)
    prob_a_v, pred_a_v = get_prediction_by_period(united_model_path,['arterial','venous'],data_path)

    prob_three, pred_three = get_prediction_by_period(united_model_path,['non_contrast','arterial','venous'],data_path)

    prob_data = pd.DataFrame({'CaseName':casename, 'label':label,
                              'NCP': prob_n, 'CMP':prob_a, 'NP':prob_v,
                              'NCP+CMP': prob_n_a, 'NCP+NP':prob_n_v, 'CMP+NP':prob_a_v,
                              'NCP+CMP+NP': prob_three})
    prob_data.to_csv(os.path.join(save_dir,save_name+'_prob.csv'),index=None)

    pred_data = pd.DataFrame({'CaseName': casename, 'label': label,
                              'NCP': pred_n, 'CMP': pred_a, 'NP': pred_v,
                              'NCP+CMP': pred_n_a, 'NCP+NP': pred_n_v, 'CMP+NP': pred_a_v,
                              'NCP+CMP+NP': pred_three})
    pred_data.to_csv(os.path.join(save_dir, save_name + '_pred.csv'), index=None)


def examine_model(prob_path, pred_path, data_path):
    prob_data = pd.read_csv(prob_path)
    pred_data = pd.read_csv(pred_path)

    label = prob_data['label'].values
    model_list = prob_data.columns[2:]

    dic = {'NCP':'non_contrast', 'CMP':'arterial', 'NP':'venous', 'NCP+CMP':['non_contrast','arterial'],
           'NCP+NP':['non_contrast','venous'], 'CMP+NP':['arterial','venous'], 'NCP+CMP+NP':['non_contrast','arterial','venous']}
    for model_name in model_list:
        period_list = dic[model_name]
        if isinstance(period_list,list):
            compute_ic(united_model_path, data_path, period_list)
        else:
            compute_period_ic(united_model_path, data_path, period_list)

        prob = prob_data[model_name]
        pred = pred_data[model_name]
        auc = roc_auc_score(label, prob)
        acc = accuracy_score(label, pred)
        sen = recall_score(label, pred)
        spe = specificity_score(label, pred)

        print('{}: auc:{:.3f},\tacc: {:.3f},\tsen: {:.3f},\tspe: {:.3f}\n'.
              format(model_name,auc,acc,sen,spe))


if __name__ == '__main__':
    united_model_path = r'F:\SHZL\model\3d\ISUP\united_model_7\lr'
    test_data_path = r'F:\SHZL\model\3d\ISUP\test_data.csv'
    train_data_path = r'F:\SHZL\model\3d\ISUP\train_data.csv'
    # wilcoxon_test(united_model_path, test_data_path)
    # compute_ic(united_model_path,test_data_path,['non_contrast','venous'])
    # compute_period_ic(united_model_path,test_data_path,'venous')
    # t_test(united_model_path,'non_contrast','arterial',test_data_path)
    # t_test_all(united_model_path,test_data_path)
    # print_result_to_csv(united_model_path,test_data_path,r'F:\SHZL\model\3d\ISUP\united_model_7\lr','test')
    # print_result_to_csv(united_model_path,train_data_path,r'F:\SHZL\model\3d\ISUP\united_model_7\lr','train')
    examine_model(r'F:\SHZL\model\3d\ISUP\united_model_7\lr\train_prob.csv',
                  r'F:\SHZL\model\3d\ISUP\united_model_7\lr\train_pred.csv', train_data_path)
