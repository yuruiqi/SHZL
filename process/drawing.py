import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import math
from MeDIT.SaveAndLoad import SaveArrayAsTIFF
from MeDIT.Visualization import MergeImageWithROI
import SimpleITK as sitk


def normalize(data):
    max = np.max(data)
    min = np.min(data)
    data = (data-min)/(max-min)
    return data


def get_roc(model_dic,set):
    if set == 'train':
        data_path = r'F:\SHZL\model\3d\ISUP\united_model_7\lr\train_prob.csv'
    elif set == 'test':
        data_path = r'F:\SHZL\model\3d\ISUP\united_model_7\lr\prob.csv'
    data = pd.read_csv(data_path)
    label = data['label'].values

    for model_name in model_dic:
        pred = data[model_name].values
        fpr, tpr, threshold = roc_curve(label, pred)
        auc_value = auc(fpr, tpr)

        if model_name == 'n+a+v':
            line = '-'
        else:
            line = '-'

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(fpr, tpr, label=model_dic[model_name]+'(AUC={:.3f})'.format(auc_value), linestyle=line)


def draw_roc(dic,set,save_path):
    get_roc(dic,set)

    # get_roc({'CMP': 'training'}, 'train')
    # get_roc({'arterial': 'testing'}, 'test')
    #
    plt.legend(loc='lower right')
    plt.savefig(save_path,format='tif',dpi=1000)
    plt.show()



def draw_heatmap():
    tick_name = ['NCP', 'CMP', 'NP', 'NCP+CMP', 'NCP+NP', 'CMP+NP', 'NCP+CMP+NP']

    data = np.array([[0,    1.54e-5,    2.94e-32,   7.83e-9,    0.0164,     0.0156,     3.05e-8],
                     [0,    0,          9.94e-15,   1.74e-24,   7.24e-12,   0.0434, 7.34e-24],
                     [0,    0,          0,          2.98e-69,   3.24e-47,   7.01e-23,   2.45e-69],
                     [0,    0,          0,          0,          4.86e-4,    3.75e-17,   0.716],
                     [0,    0,          0,          0,          0,          7.001e-7,   1.70e-16],
                     [0,    0,          0,          0,          0,          0,          1.47e-3],
                     [0,    0,          0,          0,          0,          0,          0]])
    data = data+data.T+np.identity(7)
    thres = 0.01
    shape = np.shape(data)
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = data[i,j]
            if x < thres:
                data[i,j] = thres

    data=np.log10(data)
    print(data)

    plt.xticks(range(shape[0]),tick_name,rotation=22.5)
    plt.yticks(range(shape[1]),tick_name)
    im = plt.imshow(data, cmap=plt.cm.jet, vmax=0.0)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(r'E:\SHZL-paper\pic\ttest_heatmap.tif',format='tif',dpi=1000)
    plt.show()


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

draw_roc({'non_contrast':'NCP', 'arterial':'CMP', 'venous':'NP'},'test',r'E:\SHZL-paper\pic\single-phase.tif')
draw_roc({'n+a':'NCP+CMP',  'n+v':'NCP+NP', 'a+v':'CMP+NP','n+a+v':'NCP+CMP+NP'},'test',r'E:\SHZL-paper\pic\multi-phase.tif')

# draw_heatmap()
