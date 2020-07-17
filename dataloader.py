import glob
import skimage.io as io
import torch
import random


def load_train_data(data_path: str):
    t1_path = glob.glob(data_path + '/*T1.img')
    t2_path = glob.glob(data_path + '/*T2.img')
    gt_path = glob.glob(data_path + '/*label.img')  # ground truth

    t1_list = []
    t2_list = []
    gt_list = []

    for i in range(len(gt_path)):
        img_t1 = t1_path[i]  # type : string
        img_t2 = t2_path[i]
        img_gt = gt_path[i]

        t1 = io.imread(img_t1, plugin='simpleitk')  # type : ndarray
        t2 = io.imread(img_t2, plugin='simpleitk')
        gt = io.imread(img_gt, plugin='simpleitk')

        t1 = torch.tensor(t1, dtype=torch.float32)  # tyep : torch.Tensor
        t2 = torch.tensor(t2, dtype=torch.float32)
        gt = torch.tensor(gt, dtype=torch.float32)

        gt[gt == 10] = 1  # CSF
        gt[gt == 150] = 2  # GM
        gt[gt == 250] = 3  # WM

        t1, t2, gt = cut_zero(t1, t2, gt)  # 去掉所有维度上全零的层
        t1_list.append(t1)
        t2_list.append(t2)
        gt_list.append(gt)

        # patch_size = 64

    print('load train data successfully')
    return t1_list, t2_list, gt_list


def cut_zero(t1, t2, gt):
    c_begin = c_end = h_begin = h_end = w_begin = w_end = 0

    for c_begin in range(gt.shape[0]):
        if gt[c_begin, :, :].sum() != 0:
            break
    for c_end in reversed(range(gt.shape[0])):
        if gt[c_end, :, :].sum() != 0:
            break

    for h_begin in range(gt.shape[1]):
        if gt[:, h_begin, :].sum() != 0:
            break
    for h_end in reversed(range(gt.shape[1])):
        if gt[:, h_end, :].sum() != 0:
            break

    for w_begin in range(gt.shape[2]):
        if gt[:, :, w_begin].sum() != 0:
            break
    for w_end in reversed(range(gt.shape[2])):
        if gt[:, :, w_end].sum() != 0:
            break

    t1 = t1[c_begin:c_end, h_begin:h_end, w_begin:w_end]
    t2 = t2[c_begin:c_end, h_begin:h_end, w_begin:w_end]
    gt = gt[c_begin:c_end, h_begin:h_end, w_begin:w_end]

    return t1, t2, gt


def random_patch(t1_list, t2_list, gt_list, patch_size):
    index = random.randint(0, len(gt_list)-1)

    c = random.randint(0, gt_list[index].shape[0] - patch_size - 1)  # ret a rand num in [a,b]
    h = random.randint(0, gt_list[index].shape[1] - patch_size - 1)
    w = random.randint(0, gt_list[index].shape[2] - patch_size - 1)

    t1 = t1_list[index][c:c + patch_size, h:h + patch_size, w:w + patch_size]
    t2 = t2_list[index][c:c + patch_size, h:h + patch_size, w:w + patch_size]
    gt = gt_list[index][c:c + patch_size, h:h + patch_size, w:w + patch_size]

    return t1, t2, gt


def make_patch(t1, t2, gt, patch_size):
    patch_t1 = []
    patch_t2 = []
    patch_gt = []
    C, H, W = gt.shape

    # 分块
    for c in range(0, C, int((C - patch_size) / 4)):
        if c + patch_size > C: break
        for h in range(0, H, int((H - patch_size) / 4)):
            if h + patch_size > H: break
            for w in range(0, W, int((W - patch_size) / 4)):
                if w + patch_size > W: break
                patch_t1.append(t1[c:c + patch_size, h:h + patch_size, w:w + patch_size])
                patch_t2.append(t2[c:c + patch_size, h:h + patch_size, w:w + patch_size])
                patch_gt.append(gt[c:c + patch_size, h:h + patch_size, w:w + patch_size])

    return patch_t1, patch_t2, patch_gt
