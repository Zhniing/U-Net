import time
import torch

bar_len = 40

begin_time = time.time()
last_time = begin_time


def progress_bar(i, n):
    """Print progress bar."""
    global begin_time, last_time
    now = time.time()
    total_time = now - begin_time
    step_time = now - last_time
    last_time = now
    p = i / n * bar_len
    a = '=' * round(p)
    b = ' ' * (bar_len - round(p))
    print('\r[%s%s] total: %s | step: %s |' % (a, b, time_format(total_time), time_format(step_time)), end='')  # 不换行


def time_format(second):
    """Transfer "second" to the format of "hour/minute/second/millisecond"."""
    # second = int(second)
    h = int(second // 3600)
    m = int(second % 3600 // 60)
    s = int(second % 60)
    ms = int(second * 1000 % 1000)
    ans = ''
    if h != 0:
        ans += str(h) + 'h'
    if m != 0:
        ans += str(m) + 'm'
    if s != 0:
        ans += str(s) + 's'
    if ans == '' and ms != 0:  # 少于1s则显示ms
        ans += str(ms) + 'ms'
    return ans


def get_model_size(model):
    """Return the number of model parameters."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    return num_params


def get_wrong(predict, gt):
    """计算出一个分割错误的mask"""
    mask = torch.zeros_like(gt)
    mask[predict != gt] = gt[predict != gt]
    return mask
