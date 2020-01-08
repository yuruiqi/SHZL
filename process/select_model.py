import numpy as np
import pandas as pd
import os
import shutil


def copy_model(original_model_dir, model_name, united_model_path, model_name_dic):
    for period in model_name_dic.keys():
        period_path = os.path.join(original_model_dir,period)
        kind_path = os.path.join(period_path, model_name_dic[period][0])
        model_dir_path = os.path.join(kind_path, model_name)

        for name in os.listdir(model_dir_path):
            if model_name_dic[period][1] in name:
                model_path = os.path.join(model_dir_path, name)

                model_save_path = os.path.join(united_model_path, period)
                model_save_path = os.path.join(model_save_path, name)
                shutil.copytree(model_path, model_save_path)


# def compute_cvval_error(model_path):
    


original_model_path = r'F:\SHZL\model\3d\ISUP'
united_model_path = r'F:\SHZL\model\3d\ISUP\united_model_7\svm'
model_name_dic = {'non_contrast':['2kind_feature','_12_SVM'], 'arterial':['4kind_feature','_6_SVM'], 'venous':['2kind_feature','_1_SVM']}
model_name = 'model_7_debugfae'

copy_model(original_model_path, model_name, united_model_path, model_name_dic)
