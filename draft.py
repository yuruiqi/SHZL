import pandas as pd
import numpy as np
import os
from MeDIT.UsualUse import Imshow3DArray
from MeDIT.SaveAndLoad import SaveArrayAsTIFF
from MeDIT.Visualization import MergeImageWithROI
import SimpleITK as sitk
import matplotlib.pyplot as plt


def normalize(data):
    max = np.max(data)
    min = np.min(data)
    data = (data-min)/(max-min)
    return data


def save_pic(phase_name,clip):
    case_path = r'F:\SHZL\data\3d\all\BAO_ZHEN_SHENG_10713516'
    save_dir = r'E:\SHZL-paper\pic'

    image_path = os.path.join(case_path,phase_name+'.nii')
    label_path = os.path.join(case_path,phase_name+'_label.nii')
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    image = normalize(np.transpose(image,[1,2,0]))
    label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
    label = normalize(np.transpose(label,[1,2,0]))

    patch=[[200,400],[300,500]]
    image = image[patch[0][0]:patch[0][1], patch[1][0]:patch[1][1], 22]
    label = label[patch[0][0]:patch[0][1], patch[1][0]:patch[1][1], 22]
    # Imshow3DArray(image)

    data = np.clip(image, clip[0], clip[1])
    data = normalize(data)
    data = MergeImageWithROI(data, label)
    save_path = os.path.join(save_dir, phase_name+'.tif')
    SaveArrayAsTIFF(data, save_path,dpi=(1000,1000))


save_pic('non_contrast',[0.34,0.41])
save_pic('arterial',[0.33,0.45])
save_pic('venous',[0.33,0.45])
