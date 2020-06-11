import glob
import skimage.io as io
import torch


def load_train_data(data_path: str):
    t1_path = glob.glob(data_path + '/*T1.img')
    t2_path = glob.glob(data_path + '/*T2.img')
    gt_path = glob.glob(data_path + '/*label.img')  # ground truth

    patch_t1 = []
    patch_t2 = []
    patch_gt = []

    for i in range(len(t1_path)):
        img_t1 = t1_path[i]  # type : string
        img_t2 = t2_path[i]
        img_gt = gt_path[i]

        t1 = io.imread(img_t1, plugin='simpleitk')  # type : ndarray
        t2 = io.imread(img_t2, plugin='simpleitk')
        gt = io.imread(img_gt, plugin='simpleitk')

        t1 = torch.tensor(t1, dtype=torch.float32)  # tyep : torch.Tensor
        t2 = torch.tensor(t2, dtype=torch.float32)
        gt = torch.tensor(gt, dtype=torch.float32)

        C, H, W = t1.shape
        patch_size = 64

        for c in range(0, C, int((C-patch_size)/4)):
            if c+patch_size > C: break
            for h in range(0, H, int((H-patch_size)/4)):
                if h + patch_size > H: break
                for w in range(0, W, int((W-patch_size)/4)):
                    if w + patch_size > W: break
                    patch_t1.append(t1[c:c+patch_size, h:h+patch_size, w:w+patch_size])
                    patch_t2.append(t2[c:c+patch_size, h:h+patch_size, w:w+patch_size])
                    patch_gt.append(gt[c:c+patch_size, h:h+patch_size, w:w+patch_size])

    print('load train data successfully')
    return patch_t1, patch_t2, patch_gt
