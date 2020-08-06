import numpy as np


def get_dice(prediction, groundtruth):
    dice = np.zeros(3)
    for i in range(len(groundtruth)):
        p = prediction[:, i+1, :, :]
        g = groundtruth[i]
        dice[i] = 2 * p.mul(g).sum() / (p.sum() + g.sum())

    return dice  # csf, gm, wm
