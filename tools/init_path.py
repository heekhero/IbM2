import os
import os.path as osp
import sys


# 定义函数add_path,将路径加入到sys.path中
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


parent_dir = osp.abspath(os.path.join(__file__, '..', '..'))  # 通过__file__找到当前路径(工程文件的路径)
print(parent_dir)
add_path(parent_dir)