import glob
import skimage.io as io
import torch
import random
import numpy as np
import SimpleITK as sitk


def load_data(data_path: str, validation=0, patch_size=64):
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
            t1, t2 = z_score(t1, t2)  # 数据归一化

            t1_list.append(t1)
            t2_list.append(t2)
            gt_list.append(gt)
        else:  # load validation set
            t1, t2 = z_score(t1, t2)  # 数据归一化
            v_t1_list.append(t1)
            v_t2_list.append(t2)
            v_gt_list.append(gt)

    print('Load data successfully')

    #  训练集: 去掉全0边的图像, 验证集: 完整图像[256,192,144]
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


def make_patch(t1_list, t2_list, gt_list, p_size):
    # 对输入list的每一个样本进行分块，返回分块结果: *_list_p[样本][某个样本的小块]
    t1_list_p = []
    t2_list_p = []
    gt_list_p = []

    for i in range(len(gt_list)):  # 对于每一个待分块样本
        tmp_t1 = []
        tmp_t2 = []
        tmp_gt = []

        t1 = t1_list[i]
        t2 = t2_list[i]
        gt = gt_list[i]

        # 分块
        C, H, W = gt.shape
        for c in range(0, C, int((C - p_size) / 4)):
            if c + p_size > C: break
            for h in range(0, H, int((H - p_size) / 4)):
                if h + p_size > H: break
                for w in range(0, W, int((W - p_size) / 4)):
                    if w + p_size > W: break
                    tmp_t1.append(t1[c:c + p_size, h:h + p_size, w:w + p_size])
                    tmp_t2.append(t2[c:c + p_size, h:h + p_size, w:w + p_size])
                    tmp_gt.append(gt[c:c + p_size, h:h + p_size, w:w + p_size])

        t1_list_p.append(tmp_t1)
        t2_list_p.append(tmp_t2)
        gt_list_p.append(tmp_gt)

    return t1_list_p, t2_list_p, gt_list_p


def fuse(patches, patch_size, shape):
    C, H, W = shape
    result = np.zeros(shape=[C, 4, H, W])
    count = np.zeros(shape=[C, 4, H, W])

    i = 0
    for c in range(0, C, int((C - patch_size) / 4)):
        if c + patch_size > C: break
        for h in range(0, H, int((H - patch_size) / 4)):
            if h + patch_size > H: break
            for w in range(0, W, int((W - patch_size) / 4)):
                if w + patch_size > W: break
                # if output_list[i].sum() != 0:
                result[c:c + patch_size, :, h:h + patch_size, w:w + patch_size] += patches[i]
                count[c:c + patch_size, :, h:h + patch_size, w:w + patch_size] += 1
                i += 1

    result = result / count  # 直接用除号？
    return torch.from_numpy(result).type(torch.float32)


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


def save_image(img, src):
    # io.imsave('./result/' + src + '.img', img, plugin='simpleitk')  # bug:#2292
    img = sitk.GetImageFromArray(img)
    path = './result/' + src + '.img'
    sitk.WriteImage(img, path)


# 数据归一化 Z-score
def z_score(t1, t2):
    mask = t1 > 0
    mean = t1[mask].mean()
    std = t1[mask].std()
    t1 = (t1 - mean) / std
    mean = t2[mask].mean()
    std = t2[mask].std()
    t2 = (t2 - mean) / std
    return t1, t2
