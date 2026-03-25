import argparse
import sys

import concurrent.futures
import tempfile
import uuid
import shutil
import os
from functools import partial
sys.path.append('../')

from core.chiplet_eva import chiplet_evaluator

def creat_sim_tmpdir():
    work_dir = f"chiplet_{uuid.uuid4().hex}"
    temp_dir = os.path.join(work_dir)
    # print(temp_dir)
    os.mkdir(temp_dir)
    os.mkdir(temp_dir+'/floorplan')
    os.mkdir(temp_dir+'/outputs')
    os.mkdir(temp_dir+'/ptrace')
    shutil.copy("../tmp/draw.sh", temp_dir)
    shutil.copy("../tmp/example.config", temp_dir)
    shutil.copy("../tmp/example.materials", temp_dir)
    shutil.copy("../tmp/run.sh", temp_dir)
    new_lcf = open(temp_dir+ '/floorplan/' + 'example.lcf', 'w')
    with open('../tmp/floorplan/example.lcf','r') as f:
        lines = f.readlines()
        for line in lines:
            if '../tmp' in line and "#" not in line:
                pathes = line.split('/')
                pathes[0] = '.'
                pathes[1] = work_dir
                new_line = '/'.join(pathes)
            else:
                new_line = line
            new_lcf.write(new_line)
    new_lcf.close()
    return work_dir


