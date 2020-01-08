import numpy as np
import pandas as pd
import os
from problem.SigValue import other_features


def ExtractFeature(feature_list, to_path):
    from_path = r'D:\PycharmProjects\learning\SHZL\feature_data\3d\all\all_features.csv'
    label_path = r'F:\SHZL\model\3d\ISUP\ISUP_feature.csv'
    all_data = pd.read_csv(from_path)
    label_data = pd.read_csv(label_path)

    label_df = pd.DataFrame({'CaseName':label_data['CaseName'].astype('str'), 'label':label_data['label']})
    df = pd.DataFrame({'CaseName':all_data['CaseName']})
    # 必须在这里计算case个数，因为之后drop并不会重新排列index，导致index超出drop后的case个数
    df_len = df.shape[0]

    to_drop = df[df['CaseName'].isin(['CHEN_XIAO_CHUN2_10838430', 'HE_LIN_XUAN2_10790012'])]
    for drop_index in to_drop.index.values.tolist():
        df.drop(index=drop_index, inplace=True)

    for i in range(df_len):
        try:
            df.loc[i, 'CaseName'] = df.loc[i, 'CaseName'][-8:]
        except KeyError:
            continue

    for feature in feature_list:
        feature_data = all_data[feature]
        df = pd.concat([df,feature_data],axis=1)

    df = pd.merge(label_df, df, how='left', on=['CaseName'])
    df.to_csv(to_path,index=None)


ExtractFeature(other_features,r'F:\SHZL\model\3d\ISUP\model_compare\other_features.csv')
