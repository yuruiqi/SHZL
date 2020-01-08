import numpy as np
import pandas as pd
import os


def seperate_period_data(dimension, standard):
    model_path = r'F:\SHZL\model'

    standard_dir = os.path.join(model_path, dimension)
    standard_dir = os.path.join(standard_dir, standard)

    selected_data_path = os.path.join(model_path,dimension)
    selected_data_path = os.path.join(selected_data_path,standard)
    selected_data_path = os.path.join(selected_data_path,standard+'_feature.csv')

    selected_non_contrast_data_path = os.path.join(standard_dir,'non_contrast\\selected_non_contrast.csv')
    selected_arterial_data_path = os.path.join(standard_dir,'arterial\\selected_arterial.csv')
    selected_venous_data_path = os.path.join(standard_dir,'venous\\selected_venous.csv')

    selected_data = pd.read_csv(selected_data_path)
    non_contrast_index = ['CaseName', 'label']
    arterial_index = ['CaseName', 'label']
    venous_index = ['CaseName', 'label']
    for index in selected_data:
        if 'non_contrast' in index:
            non_contrast_index.append(index)
        elif 'arterial' in index:
            arterial_index.append(index)
        elif 'venous' in index:
            venous_index.append(index)

    selected_non_contrast_data = pd.DataFrame(selected_data[non_contrast_index])
    selected_arterial_data = pd.DataFrame(selected_data[arterial_index])
    selected_venous_data = pd.DataFrame(selected_data[venous_index])

    selected_non_contrast_data.to_csv(selected_non_contrast_data_path, index=None)
    selected_arterial_data.to_csv(selected_arterial_data_path, index=None)
    selected_venous_data.to_csv(selected_venous_data_path, index=None)


standard_list = ['ISUP','furman']
seperate_period_data('3d', standard_list[0])
