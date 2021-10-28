# -*- coding: utf-8 -*-
# @File    : project_create.py
# @Author  : Hua Guo
# @Time    : 2021/10/28 上午10:13
# @Disc    :
import subprocess as sp
import os

# ================== Config =================
dirs = ['data', 'notebooks', 'model_finished', 'model_training', 'logs', 'report', 'scripts']

for directory in dirs:
    cmd1 = f"mkdir {directory}"
    gitkeep = os.path.join(directory, '.gitkeep')
    cmd2 = f"touch {gitkeep}"
    for cmd in [cmd1, cmd2]:
        print(cmd)
        sp.call(cmd, shell=True)
# ============================================