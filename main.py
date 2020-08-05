from network import *
from dataloader import *
from evaluation import *
from utility import *
import torch.optim as optim
from lovasz_losses import lovasz_softmax

gpu_idx = 0
cuda = torch.device(gpu_idx)

val_idx = 0  # validation index
data_path = '../dataset/iseg2017/'
t1_list, t2_list, gt_list, v_t1_list, v_t2_list, v_gt_list = \
    load_train_data(data_path, val_idx)  # train & validation set

model = Unet(in_ch=2, out_ch=4).cuda(gpu_idx)
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters())

iterations = 150  # 一个epoch中的迭代次数
for epoch in range(100):
    print('epoch', epoch + 1)
    total_loss = 0
    csf_dice = 0
    gm_dice = 0
    wm_dice = 0
    patch_size = 64

    # train step
    for i in range(iterations):
        t1, t2, gt = random_patch(t1_list, t2_list, gt_list, patch_size)

        t1 = t1.unsqueeze(dim=1)  # 增加一个维度
        t2 = t2.unsqueeze(dim=1)

        image = torch.cat((t1, t2), dim=1).to(gpu_idx)
        gt = gt.cuda(gpu_idx)

        output = model(image)

        # Dice
        csf_gt = torch.zeros_like(gt)
        gm_gt = torch.zeros_like(gt)
        wm_gt = torch.zeros_like(gt)
        csf_gt[gt == 1] = 1
        gm_gt[gt == 2] = 1
        wm_gt[gt == 3] = 1
        csf_dice += get_dice(output[:, 1, :, :], csf_gt)
        gm_dice += get_dice(output[:, 2, :, :], gm_gt)
        wm_dice += get_dice(output[:, 3, :, :], wm_gt)

        loss = lovasz_softmax(output, gt)
        total_loss += loss

        # 打印结果
        progress_bar(i, iterations)
        print(' %3d/%3d | loss:%.3f | csf:%.3f | gm:%.3f | wm:%.3f'
              % (i + 1, iterations,
                 total_loss.item() / (i + 1),
                 csf_dice.item() / (i + 1),
                 gm_dice.item() / (i + 1),
                 wm_dice.item() / (i + 1)),
              end='')

        loss.backward()  # Error Backpropagation for computing gradients
        optimizer.step()  # Update parameters, base on the gradients
        optimizer.zero_grad()

    # validation step
    print("\nValidate:", end='')
    for i in range(len(v_gt_list)):
        t1_patchs, t2_patchs, gt_patchs = \
            make_patch(v_t1_list[i], v_t2_list[i], v_gt_list[i], patch_size)
        output_list = []

        for j in range(len(gt_patchs)):
            t1 = t1_patchs[j]
            t2 = t2_patchs[j]
            gt = gt_patchs[j]

            t1 = t1.unsqueeze(dim=1)  # 增加一个维度
            t2 = t2.unsqueeze(dim=1)

            image = torch.cat((t1, t2), dim=1).cuda(gpu_idx)
            gt = gt.cuda(gpu_idx)

            output = model(image)
            output_list.append(output.cpu().detach().numpy())  # detach()脱离梯度计算,可用.data代替

        prediction = fuse(output_list, patch_size, v_gt_list[i].shape)
        groundtruth = v_gt_list[i]
        loss = lovasz_softmax(prediction, groundtruth)

        # Dice
        csf_dice = 0
        gm_dice = 0
        wm_dice = 0
        csf_gt = torch.zeros_like(groundtruth)
        gm_gt = torch.zeros_like(groundtruth)
        wm_gt = torch.zeros_like(groundtruth)
        csf_gt[groundtruth == 1] = 1
        gm_gt[groundtruth == 2] = 1
        wm_gt[groundtruth == 3] = 1
        csf_dice += get_dice(prediction[:, 1, :, :], csf_gt)
        gm_dice += get_dice(prediction[:, 2, :, :], gm_gt)
        wm_dice += get_dice(prediction[:, 3, :, :], wm_gt)

        print(' loss:%.3f | csf:%.3f | gm:%.3f | wm:%.3f'
              % (loss.item() / (i + 1),
                 csf_dice.item() / (i + 1),
                 gm_dice.item() / (i + 1),
                 wm_dice.item() / (i + 1)),
              end='')

    print()  # 换行
