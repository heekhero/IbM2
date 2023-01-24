import os
import socket

HOST_NAME = socket.getfqdn(socket.gethostname())

if 'ubuntu' in HOST_NAME:
    prefix = 'root'
else:
    prefix = 'opt'

DATA3_ROOT_PATH = '/mnt/data3/fumh/FSL/BSearch/RealFewShot_new_no_cyan'
PATH = os.path.dirname(os.path.realpath(__file__))
# DATA_PATH = os.path.join(os.path.abspath(os.path.join(__file__, '..', '..', '..', '..')), 'datasets')
DATA_PATH = '/mnt/data3/fumh/datasets'
SPLIT_PATH = os.path.join(DATA_PATH, 'few_shot_split')

EPSILON = 1e-8

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')


