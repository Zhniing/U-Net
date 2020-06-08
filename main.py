from network import *
import torch
from dataloader import *

data_path = '../datasets/iseg2017/'
for filename in glob.glob(data_path + '/*T1.img'):
    print(filename)

print("hello")