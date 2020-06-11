import torch


def dice(prediction, groundtruth):
    csf_gt = torch.zeros_like(groundtruth)
    gm_gt = torch.zeros_like(groundtruth)
    wm_gt = torch.zeros_like(groundtruth)

    csf_gt[groundtruth == 1] = 1
    gm_gt[groundtruth == 2] = 1
    wm_gt[groundtruth == 3] = 1

    P = prediction.sum()

    csf_dice = 2 * prediction.mul(csf_gt) / (P + groundtruth.sum())
    gm_dice = 2 * prediction.mul(gm_gt) / (P + groundtruth.sum())
    wm_dice = 2 * prediction.mul(wm_gt) / (P + groundtruth.sum())

    return csf_dice, gm_dice, wm_dice
