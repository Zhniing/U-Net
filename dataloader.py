import glob
import skimage.io as io
import torch
import random
import numpy as np


def load_data(data_path: str, validation=0):
    t1_path = glob.glob(data_path + '/*T1.img')
    t2_path = glob.glob(data_path + '/*T2.img')
    gt_path = glob.glob(data_path + '/*label.img')  # ground truth
    t1_path.sort()
    t2_path.sort()
    gt_path.sort()

    # train set
    t1_list = []
    t2_list = []
    gt_list = []
    # validation set
    v_t1_list = []
    v_t2_list = []
    v_gt_list = []

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

        if i != validation:  # load train set
            t1, t2, gt = cut_zero(t1, t2, gt)  # 去掉所有维度上全零的层
            t1_list.append(t1)
            t2_list.append(t2)
            gt_list.append(gt)
        else:  # load validation set
            patches_t1, patches_t2 = make_patch(t1, t2, 64)  # todo: 换成patch_size变量
            v_t1_list.append(patches_t1)
            v_t2_list.append(patches_t2)
            v_gt_list.append(gt)

    print('load train data successfully')
    return t1_list, t2_list, gt_list, v_t1_list, v_t2_list, v_gt_list


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
    index = random.randint(0, len(gt_list) - 1)

    c = random.randint(0, gt_list[index].shape[0] - patch_size - 1)  # ret a rand num in [a,b]
    h = random.randint(0, gt_list[index].shape[1] - patch_size - 1)
    w = random.randint(0, gt_list[index].shape[2] - patch_size - 1)

    t1 = t1_list[index][c:c + patch_size, h:h + patch_size, w:w + patch_size]
    t2 = t2_list[index][c:c + patch_size, h:h + patch_size, w:w + patch_size]
    gt = gt_list[index][c:c + patch_size, h:h + patch_size, w:w + patch_size]

    return t1, t2, gt


def make_patch(t1, t2, patch_size):
    patches_t1 = []
    patches_t2 = []
    # patches_gt = []
    C, H, W = t1.shape

    # 分块
    for c in range(0, C, int((C - patch_size) / 4)):
        if c + patch_size > C: break
        for h in range(0, H, int((H - patch_size) / 4)):
            if h + patch_size > H: break
            for w in range(0, W, int((W - patch_size) / 4)):
                if w + patch_size > W: break
                patches_t1.append(t1[c:c + patch_size, h:h + patch_size, w:w + patch_size])
                patches_t2.append(t2[c:c + patch_size, h:h + patch_size, w:w + patch_size])
                # patches_gt.append(gt[c:c + patch_size, h:h + patch_size, w:w + patch_size])

    return patches_t1, patches_t2


def fuse(output_list, patch_size, shape):
    i = 0
    while output_list[i].sum() == 0:
        i += 1
    C, H, W = shape
    prediction = np.zeros(shape=[C, 4, H, W])
    count = np.zeros(shape=[C, 4, H, W])

    i = 0
    for c in range(0, C, int((C - patch_size) / 4)):
        if c + patch_size > C: break
        for h in range(0, H, int((H - patch_size) / 4)):
            if h + patch_size > H: break
            for w in range(0, W, int((W - patch_size) / 4)):
                if w + patch_size > W: break
                # if output_list[i].sum() != 0:
                prediction[c:c + patch_size, :, h:h + patch_size, w:w + patch_size] += output_list[i]
                count[c:c + patch_size, :, h:h + patch_size, w:w + patch_size] += 1
                i += 1

    prediction = prediction / count  # 直接用除号？
    return torch.Tensor(prediction)


def make_image(t1, t2):
    t1 = t1.unsqueeze(dim=1)  # 增加一个维度
    t2 = t2.unsqueeze(dim=1)
    return torch.cat((t1, t2), dim=1)


def split_gt(gt):
    csf_gt = torch.zeros_like(gt)
    gm_gt = torch.zeros_like(gt)
    wm_gt = torch.zeros_like(gt)
    csf_gt[gt == 1] = 1
    gm_gt[gt == 2] = 1
    wm_gt[gt == 3] = 1

    return csf_gt, gm_gt, wm_gt
