import os

def rename_files(directory):
    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否以_gt.nii.gz结尾
        if filename.endswith('_img.nii.gz'):
            # 构造新文件名，将_gt替换为_img
            new_filename = filename.replace('_img.nii.gz', '_pred.nii.gz')
            # 获取完整的源文件和目标文件路径
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {filename} -> {new_filename}')

# 指定要处理的文件夹路径
folder_path = './pictures/Synapse/Source'
rename_files(folder_path)
