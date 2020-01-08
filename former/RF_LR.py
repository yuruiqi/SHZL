import pandas as pd
import  numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score

import os


def generate_pred_csv(period,best_model,trainortest):
    path = 'E:\\study\\postgraduate\\mission\\20190717\\RF_LR\\data\\' + period + '\\result\\' + best_model

    if trainortest == 'train':
        f_perdict = pd.read_csv(path+'\\all_train_info.csv')
    elif trainortest == 'test':
        f_perdict = pd.read_csv(path + '\\test_info.csv')
    else:
        print('error:generate_pred_csv')

    casename = f_perdict['CaseName']
    pred = f_perdict['Pred']
    label = f_perdict['Label']

    f_out = pd.DataFrame({'CaseName':casename, 'Label':label, 'Pred_'+period:pred},index= None)
    f_out_path = 'E:\\study\\postgraduate\\mission\\20190717\\RF_LR\\set\\'+ trainortest + '\\pred_' + period +'.csv'
    f_out.to_csv(f_out_path,index=None)


def get_all_pred(trainortest):

    fold_path = 'E:\\study\\postgraduate\\mission\\20190717\\RF_LR\set\\' + trainortest

    noncontrast_data = pd.read_csv(fold_path+'\\pred_non_contrast.csv')
    arterial_data = pd.read_csv(fold_path+'\\pred_arterial.csv')
    venous_data = pd.read_csv(fold_path+'\\pred_venous.csv')

    all_data = noncontrast_data.merge(arterial_data)
    all_data = all_data.merge(venous_data)
    all_data = all_data.drop_duplicates()

    all_data.to_csv(fold_path + '\\pred_all.csv', index=None)


def get_lr_model():
    lr = LogisticRegression(solver='liblinear')

    data = pd.read_csv(r'E:\study\postgraduate\mission\20190717\RF_LR\set\train\pred_all.csv')
    label = data['Label'].values
    features = data[['Pred_non_contrast', 'Pred_arterial', 'Pred_venous']].values

    lr.fit(features,label)
    y_train = lr.predict(features)
    y_train_prob = lr.predict_proba(features)[:,1]

    acc = accuracy_score(label,y_train)
    auc = roc_auc_score(label,y_train_prob)
    print(lr.coef_,lr.intercept_)
    print('train:',acc,auc)

    return lr


def get_one_set(period,trainortest):
    if trainortest == 'train':
        set_noncontrast= pd.read_csv(r'E:\study\postgraduate\mission\20190717\RF_LR\data\non_contrast\set\train_numeric_feature.csv')
    elif trainortest == 'test':
        set_noncontrast= pd.read_csv(r'E:\study\postgraduate\mission\20190717\RF_LR\data\non_contrast\set\test_numeric_feature.csv')
    else:
        print('error:get_one_set')

    case = list(set_noncontrast['Unnamed: 0'])

    if period == 'arterial':
        feature = pd.read_csv(r'E:\study\postgraduate\mission\20190717\RF_LR\data\arterial\data_arterial_original.csv')
    else:
        feature = pd.read_csv(r'E:\study\postgraduate\mission\20190717\RF_LR\data\venous\data_venous_original.csv')

    set = pd.DataFrame()
    for i in range(len(case)):
        row = feature.loc[feature['CaseName'] == case[i]]
        if len(row['CaseName']) > 1:
            set= set.append(row[0:1], ignore_index=True)
        else:
            set = set.append(row, ignore_index=True)

    save_path = 'E:\\study\\postgraduate\\mission\\20190717\\RF_LR\\data\\' + period + \
                '\\set\\' + trainortest + '_numeric_feature.csv'
    print(save_path)
    set.to_csv(save_path, index=None)


def from_noncontrast_getset():
    get_one_set('arterial' , 'train')
    get_one_set('arterial' , 'test')
    get_one_set('venous' , 'train')
    get_one_set('venous' , 'test')


# from_noncontrast_getset()

# non_contrast_model = 'Norm0Center_PCA_ANOVA_12_RF'
# arterial_model = 'Norm0Center_PCC_ANOVA_3_RF'
# venous_model = 'Norm0Center_PCA_ANOVA_8_RF'
#
# generate_pred_csv('non_contrast',non_contrast_model,'train')
# # generate_pred_csv('non_contrast',non_contrast_model,'test')
# # generate_pred_csv('arterial',arterial_model,'train')
# # generate_pred_csv('arterial',arterial_model,'test')
# # generate_pred_csv('venous',venous_model,'train')
# # generate_pred_csv('venous',venous_model,'test')
#
# get_all_pred('train')
# get_all_pred('test')

lr = get_lr_model()
test_data = pd.read_csv(r'E:\study\postgraduate\mission\20190717\RF_LR\set\test\pred_all.csv')
label = test_data['Label'].values
features = test_data[['Pred_non_contrast', 'Pred_arterial', 'Pred_venous']].values

y_test = lr.predict(features)
y_test_prob = lr.predict_proba(features)[:,1]

acc = accuracy_score(label,y_test)
auc = roc_auc_score(label,y_test_prob)
print('test:',acc,auc)
