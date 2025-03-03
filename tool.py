import os
import random
from shutil import move

# 指定源文件夹和目标文件夹
source_dir = r'C:\Users\FullZero\Desktop\bs\CSGO2\train'
test_dir = r'C:\Users\FullZero\Desktop\bs\CSGO2\test'

# 指定图片和标签文件夹
image_dir = os.path.join(source_dir, 'images')
label_dir = os.path.join(source_dir, 'labels')

# 确保目标文件夹存在
os.makedirs(test_dir, exist_ok=True)
os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'labels'), exist_ok=True)

# 获取所有的图像和标签文件
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
label_files = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]

# 创建一个字典来存储文件名和其对应的后缀
files_dict = {}

# 分离文件名和后缀，并将它们存储在字典中
for file in image_files:
    # 分离文件名和扩展名
    base_name = os.path.splitext(file)[0]
    if base_name in files_dict:
        files_dict[base_name]['image'] = file
    else:
        files_dict[base_name] = {'image': file}

for file in label_files:
    # 分离文件名和扩展名
    base_name = os.path.splitext(file)[0]
    if base_name in files_dict:
        files_dict[base_name]['label'] = file
    else:
        files_dict[base_name] = {'label': file}

# 确保所有图像都有对应的标签
assert all('image' in d and 'label' in d for d in files_dict.values())

# 随机选择10%的文件
num_files_to_select = int(len(files_dict) * 0.1)
selected_files = random.sample(list(files_dict.keys()), num_files_to_select)

# 移动选中的文件到测试集文件夹
for file_base_name in selected_files:
    # 移动图像文件
    src_image_path = os.path.join(image_dir, files_dict[file_base_name]['image'])
    dst_image_path = os.path.join(test_dir, 'images', files_dict[file_base_name]['image'])
    move(src_image_path, dst_image_path)

    # 移动标签文件
    src_label_path = os.path.join(label_dir, files_dict[file_base_name]['label'])
    dst_label_path = os.path.join(test_dir, 'labels', files_dict[file_base_name]['label'])
    move(src_label_path, dst_label_path)

print(f"Moved {num_files_to_select} pairs of image and label files to the test set.")
