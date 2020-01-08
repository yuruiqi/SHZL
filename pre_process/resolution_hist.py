import os
import SimpleITK as sitk
import matplotlib.pyplot as plt


dir = r'F:\SHZL\data\3d\all'
case_list = os.listdir(dir)
ratio_list = [[],[],[]]

for case in case_list:
    case_path = os.path.join(dir,case)
    arterial_path = os.path.join(case_path,'arterial.nii')
    # print(case_list)
    image = sitk.ReadImage(arterial_path)
    ratio = image.GetSpacing()
    for i in range(len(ratio)):
        ratio_list[i].append(ratio[i])

plt.hist(ratio_list[0])
plt.show()
plt.hist(ratio_list[1])
plt.show()
plt.hist(ratio_list[2])
plt.show()
