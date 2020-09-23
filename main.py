from network import *
from dataloader import *
from evaluation import *
from utility import *
import torch.optim as optim
from lovasz_losses import lovasz_softmax
import argparse
from tensorboardX import SummaryWriter
import torchvision

# torchvision.models.alexnet()

parser = argparse.ArgumentParser()
parser.add_argument('gpu_id', nargs='?', default=0, type=int, help='Assign gpu id')
parser.add_argument('log_file', nargs='?', default='', type=str, help='Assign log file name of tensorboard')
args = parser.parse_args()

gpu = args.gpu_id
cuda = torch.device(gpu)

val_idx = 0  # validation index
patch_size = 64
data_path = '../dataset/iseg2017/'
t1_list, t2_list, gt_list, v_t1_list, v_t2_list, v_gt_list = \
    load_data(data_path, val_idx, patch_size)  # train & validation set

model = Unet(in_ch=2, out_ch=4).cuda(gpu)
Focal = FocalLoss2d()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.0005)

# 是否写入tensorboard记录
log_file = args.log_file
if log_file != 'nolog':
    log_on = True
    if log_file != '':
        log_file = './runs/' + log_file
    write = SummaryWriter(log_file)
else:
    write = 0
    log_on = False

# tensorboardX可视化网络结构
# dummy_input = (torch.rand(64, 2, 64, 64).type(torch.float32).cuda(gpu),)
# write.add_graph(model, dummy_input)

iterations = 150  # 一个epoch中的迭代次数
for epoch in range(100):
    print('epoch', epoch)
    total_loss = 0
    dice = np.zeros(3)
    prediction = 0

    # train step--------------------------------------------------------
    model.train()
    for i in range(iterations):
        t1_p, t2_p, gt_p = random_patch(t1_list, t2_list, gt_list, patch_size)

        image = make_image(t1_p, t2_p).cuda(gpu)
        gt_p = gt_p.cuda(gpu)

        output = model(image)
        _, prediction = torch.max(output, dim=1)

        # Dice
        gts_p = split_gt(gt_p)
        dice += get_dice(output, gts_p)

        loss_p = lovasz_softmax(output, gt_p)
        # loss += Focal(output, gt, class_weight=[0.1, 0.4, 0.2, 0.3])
        total_loss += loss_p.item()

        # 打印结果
        progress_bar(i, iterations)
        print(' %3d/%3d | loss:%.3f | csf:%.3f | gm:%.3f | wm:%.3f'
              % (i + 1, iterations,
                 total_loss / (i + 1),
                 dice[0] / (i + 1),
                 dice[1] / (i + 1),
                 dice[2] / (i + 1)),
              end='')

        loss_p.backward()  # Error Backpropagation for computing gradients
        optimizer.step()  # Update parameters, base on the gradients
        optimizer.zero_grad()

    # Save the last train result
    save_image(prediction.cpu().numpy().astype(np.uint8), 'train')  # 保存最后一个训练（块）

    if log_on:
        # Write train loss log
        write.add_scalar('Loss/train', total_loss / 150, epoch)
        write.add_scalar('Dice_train/CSF', dice[0] / 150, epoch)
        write.add_scalar('Dice_train/GM', dice[1] / 150, epoch)
        write.add_scalar('Dice_train/WM', dice[2] / 150, epoch)
        write.add_scalar('Dice_train/mean', dice.mean() / 150, epoch)

    # validation step-------------------------------------------------------
    model.eval()  # 用了Batch Normalization就一定要区分训练和验证过程，因为验证和训练时BN的计算方式不同
    with torch.no_grad():  # Reduce memory consumption for computations with 'requires_grad=True'
        total_loss = 0
        for i in range(len(v_gt_list)):  # 有多个验证样本
            loss = 0
            out_list_p = []

            v_t1_list_p, v_t2_list_p, v_gt_list_p = \
                make_patch(v_t1_list, v_t2_list, v_gt_list, patch_size)

            for j in range(len(v_gt_list_p[i])):  # 每个验证样本 第i个验证样本的第j块
                t1_p = v_t1_list_p[i][j]
                t2_p = v_t2_list_p[i][j]
                gt_p = v_gt_list_p[i][j].cuda(gpu)

                if gt_p.sum() != 0:  # 非纯背景块
                    image = make_image(t1_p, t2_p).cuda(gpu)

                    out_p = model(image)  # out_p: 每一小块的分割结果

                    # 每块Loss
                    loss_p = lovasz_softmax(out_p, gt_p)
                    loss += loss_p  # 一个样本的总loss

                    # 保存每个块不能放在gpu里，会爆显存
                    out_list_p.append(out_p.cpu().detach().numpy())  # .detach()脱离梯度计算,不建议用.data
                else:  # 纯背景块如何处理？
                    out_list_p.append(torch.zeros_like(output).cpu().numpy())  # 直接压入全0块

            output = fuse(out_list_p, patch_size, [256, 192, 144]).cuda(gpu)  # output: 拼成完整图像的结果
            _, prediction = torch.max(output, dim=1)  # 将预测概率值转变为预测结果: [0, 1, 2, 3]
            total_loss += loss / (j + 1)

            # 整体Dice
            v_gt = v_gt_list[i].cuda(gpu)
            v_gts = split_gt(v_gt)
            dice = get_dice(output, v_gts)

            print('\nval%d: loss:%.3f | csf:%.3f | gm:%.3f | wm:%.3f'
                  % (i, loss / (j + 1),
                     dice[0], dice[1], dice[2]),
                  end='')

        print()  # 换行

        # Save the last validation result
        save_image(prediction.cpu().numpy().astype(np.uint8), 'validation')  # 保存最后一个预测（图）

        if log_on:
            # Write validation loss log
            write.add_scalar('Loss/val', total_loss / (i + 1), epoch)
            write.add_scalar('Dice_val/CSF', dice[0] / (i + 1), epoch)
            write.add_scalar('Dice_val/GM', dice[1] / (i + 1), epoch)
            write.add_scalar('Dice_val/WM', dice[2] / (i + 1), epoch)
            write.add_scalar('Dice_val/mean', dice.mean() / (i+1), epoch)

    if log_on:
        write.flush()

if log_on:
    write.close()
