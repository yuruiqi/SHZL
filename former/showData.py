from MeDIT.Visualization import Imshow3DArray
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def normalize(data):
    mx = data.max()
    mn = data.min()
    data = (data-mn)/(mx-mn)
    return data


data_path = sitk.ReadImage(r'F:\SHZL\data\3d\all\WEN_XIAO_HUI_10710409\6 artery  5.0  B30f.nii')
label_path = sitk.ReadImage(r'F:\SHZL\data\3d\all\WEN_XIAO_HUI_10710409\6 artery  5.0  B30f-label.nii')
data_nii = sitk.GetArrayFromImage(data_path)
label_nii = sitk.GetArrayFromImage(label_path)

data_nii_trans = np.transpose(data_nii,(1,2,0))
label_nii_trans = np.transpose(label_nii,(1,2,0))
data_nii_trans.astype('float64')
label_nii_trans.astype('float64')

data = normalize(data_nii_trans)
label = normalize(label_nii_trans)

print(type(data),data.dtype,data.shape)
# print(type(label),label.dtype,label.shape)

Imshow3DArray(data, label)
# Imshow3DArray(data)
