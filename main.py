from network import *
from dataloader import *
from evaluation import *
from utility import *
import torch.optim as optim
from lovasz_losses import lovasz_softmax

gpu_idx = 2

data_path = '../dataset/iseg2017/'
t1_list, t2_list, gt_list = load_train_data(data_path)

model = Unet(in_ch=2, out_ch=4).cuda(gpu_idx)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # todo 2: 采用adam优化器

iterations = 150  # 一个epoch中的迭代次数
for epoch in range(200):
    print('epoch', epoch)
    total_loss = 0
    csf_dice = 0
    gm_dice = 0
    wm_dice = 0

    for i in range(iterations):
        # t1 = t1_list[i]
        # t2 = t2_list[i]
        # gt = gt_list[i].cuda(gpu_idx)  # value: 0, 1, 2, 3
        patch_size = 64

        t1, t2, gt = random_patch(t1_list, t2_list, gt_list, patch_size)
        gt = gt.cuda(gpu_idx)

        t1 = t1.unsqueeze(dim=1)
        t2 = t2.unsqueeze(dim=1)
        # gt = gt.unsqueeze(dim=1)

        image = torch.cat((t1, t2), dim=1).cuda(gpu_idx)
        # print(torch.cuda.max_memory_allocated())

        output = model(image)

        # Dice
        csf_gt = torch.zeros_like(gt)
        gm_gt  = torch.zeros_like(gt)
        wm_gt  = torch.zeros_like(gt)
        csf_gt[gt == 1] = 1
        gm_gt[gt == 2]  = 1
        wm_gt[gt == 3]  = 1
        csf_dice += get_dice(output[:, 1, :, :], csf_gt)
        gm_dice += get_dice(output[:, 2, :, :], gm_gt)
        wm_dice += get_dice(output[:, 3, :, :], wm_gt)

        # todo: lovasz_softmax的原理，接收什么样的参数(0~1的预测概率值)，是否需要对output或gt进行额外操作？
        loss = lovasz_softmax(output, gt)
        total_loss += loss

        # 打印结果
        progress_bar(i, iterations)
        print('%3d/%3d | loss:%.3f | csf:%.3f | gm:%.3f | wm:%.3f'
                % (i + 1, iterations, total_loss.item() / (i + 1),
                   csf_dice.item() / (i+1), gm_dice.item() / (i+1), wm_dice.item() / (i+1)), end='')

        loss.backward()  # Error Backpropagation for computing gradients
        optimizer.step()  # Update parameters, base on the gradients
        optimizer.zero_grad()

        # torch.cuda.empty_cache()
    print()  # 换行
