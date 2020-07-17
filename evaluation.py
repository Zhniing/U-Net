import torch


def get_dice(prediction, groundtruth):
    dice = 2 * prediction.mul(groundtruth).sum() / (prediction.sum() + groundtruth.sum())

    return dice
