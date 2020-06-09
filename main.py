from network import *
import torch
from dataloader import *

data_path = '../datasets/iseg2017/'
list_t1, list_t2, list_gt = load_train_data(data_path)

for i in range(len(list_t1)):
    t1 = list_t1[i]
    t2 = list_t2[i]
    gt = list_gt[i]

    t1 = t1.unsqueeze(dim=1)
    t2 = t2.unsqueeze(dim=1)

    img = torch.cat((t1, t2), dim=1).to('cuda')

    U_net = Unet(in_ch=2, out_ch=4).to('cuda')
    out = U_net(img)
