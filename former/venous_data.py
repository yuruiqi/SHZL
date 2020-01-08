import pandas as pd

f=pd.read_csv(r'D:\SHZL_data\dataWithLabel_280.csv')
# f_new = pd.read_csv(r'D:\SHZL_data\data_venous.csv')


col_save=['CaseName','label']
for index in f:
    if ('venous' in index) and ('original' in index):
        col_save.append(index)

print(col_save)
f.to_csv(r'D:\SHZL_data\data_venous_original.csv', columns=col_save,index=False)

