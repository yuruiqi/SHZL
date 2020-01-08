import numpy as np
import pandas as pd
import os
import random

def generate_data(save_dir,seed):
    case_num = 100
    random.seed(seed)
    np.random.seed(seed)

    label = np.zeros(case_num)
    for i in range(case_num):
        label[i] = random.choice([0,1])

    x1 = np.random.normal(label, 0.02)
    x2 = np.random.normal(label, 0.00)
    x3 = np.random.normal(label, 1)

    data = pd.DataFrame({'label':label, 'x1':x1, 'x2':x2, 'x3':x3})
    data.to_csv(os.path.join(save_dir,'train_data.csv'))

generate_data(r'D:\PycharmProjects\test\data',19970516)
