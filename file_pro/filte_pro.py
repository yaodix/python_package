'''
Descripttion: 从根目录中提取
version: 
Author: liuYao
Date: 2020-10-13 16:11:42
LastEditors: liuYao
LastEditTime: 2020-12-31 11:50:00
'''

import os
from pathlib import Path
from glob import glob

def get_all_img_file(root_dir, file_extension = ["jpg", "png"]):
    """获取所有图片文件

    Args:
        root_dir: 文件根目录
        file_extension：文件扩展名，默认["jpg", "png"]

    Returns:
        返回指定扩展名的所有图片列表

    """
    result = []
    for ext in file_ext:
        result += [y for x in os.walk(root_dir) for y in glob(os.path.join(x[0], '*.'+ext))]

    return result

def get_all_file(root_dir : str, file_extension : list):
    """获取指定扩展名文件

    Args:
        root_dir: 文件根目录
        file_extension：文件扩展名

    Returns:
        返回指定扩展名的所有文件列表
        
    """
    result = []
    for ext in file_ext:
        # result += glob(os.path.join(root_dir, '*.'+ext))
        result += [y for x in os.walk(root_dir) for y in glob(os.path.join(x[0], '*.'+ext))]

    return result

if __name__ == "__main__":

    root_dir = "D:\\snapTemp\\1号机全系列0821"
    res = get_all_img_file(root_dir)

    pass