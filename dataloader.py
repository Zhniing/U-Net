import glob
import skimage.io as io
import torch


def load_train_data(data_path):
    t1_path = glob.glob(data_path + '/*T1.img')
    t2_path = glob.glob(data_path + '/*T2.img')
    gt_path = glob.glob(data_path + '/*label.img')  # ground truth

    list_t1 = []
    list_t2 = []
    list_gt = []

    for i in range(len(t1_path)):
        img_t1 = t1_path[i]  # type : string
        img_t2 = t2_path[i]
        img_gt = gt_path[i]

        t1 = io.imread(img_t1, plugin='simpleitk')  # type : ndarray
        t2 = io.imread(img_t2, plugin='simpleitk')
        gt = io.imread(img_gt, plugin='simpleitk')

        t1 = torch.tensor(t1, dtype=torch.float32)  # tyep : Tensor
        t2 = torch.tensor(t2, dtype=torch.float32)
        gt = torch.tensor(gt, dtype=torch.float32)

        list_t1.append(t1)  # type : list of Tensor
        list_t2.append(t2)
        list_gt.append(gt)

    print('load train data successfully')
    return list_t1, list_t2, list_gt
