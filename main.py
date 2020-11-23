from network import *
from dataloader import *
from evaluation import *
from utility import *
from loss import *
import torch.optim as optim
from lovasz_losses import lovasz_softmax
import argparse
from tensorboardX import SummaryWriter
import torchvision

# torchvision.models.alexnet()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', '-gpu', default=0, type=int, help='Assign gpu id (default: 0)')
parser.add_argument('--log_file', '-log',  default='nolog', type=str,
                    help='Assign log file name of tensorboardX (default: no log)')

args = parser.parse_args()

gpu = args.gpu_id
# cuda = torch.device(gpu)
torch.cuda.set_device(gpu)

val_idx = 0  # validation index 对比实验要固定验证集
patch_size = 64
data_path = '../dataset/iseg2017/'
t1_list, t2_list, gt_list, v_t1_list, v_t2_list, v_gt_list = \
    load_data(data_path, val_idx, patch_size)  # train & validation set

model = Unet2(in_ch=2, out_ch=4).cuda(gpu)
print("The number of parameters:", get_model_size(model))
Focal = FocalLoss2d()
soft_dice = SoftDiceLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.0005)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# 是否写入tensorboard记录
log_file = args.log_file
if log_file != 'nolog':
    log_on = True
    if log_file != '':
        log_file = './runs/' + log_file + '/'
    write = SummaryWriter(log_file)
else:
    write = 0
    log_on = False

# tensorboardX可视化网络结构
if log_on:
    # dummy_input1 = (torch.rand(64, 1, 64, 64).type(torch.float32).cuda(gpu),
    #                 torch.rand(64, 1, 64, 64).type(torch.float32).cuda(gpu))
    dummy_input1 = (torch.rand(64, 2, 64, 64).type(torch.float32).cuda(gpu),)
    write.add_graph(model, dummy_input1)

iterations = 150  # 一个epoch中的迭代次数
n_epoch = 100
lr = 0.0002

best_dice = np.zeros(3)  # 最高的验证集 mean Dice
save_flag = False

for epoch in range(n_epoch):
    print('epoch', epoch)
    total_loss = 0
    dice = np.zeros(3)
    predict = 0

    # train step--------------------------------------------------------
    model.train()
    for i in range(iterations):
        t1_p, t2_p, gt_p = random_patch(t1_list, t2_list, gt_list, patch_size)

        image = make_image(t1_p, t2_p).cuda(gpu)
        # t1_p = t1_p.unsqueeze(dim=1).cuda(gpu)
        # t2_p = t2_p.unsqueeze(dim=1).cuda(gpu)
        gt_p = gt_p.cuda(gpu)

        optimizer.zero_grad()
        output = model(image)
        # output = model(t1_p, t2_p)
        _, predict = torch.max(output, dim=1)

        # Dice
        gts_p = split_gt(gt_p)
        dice += get_dice(output, gts_p)

        # Loss
        loss_p = 0
        loss_p += lovasz_softmax(output, gt_p)
        loss_p += Focal(output, gt_p, class_weight=[0.1, 0.4, 0.2, 0.3])  # 类别权重 是按 类别数量(面积) 来设置的
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

    # lr decay
    if (epoch + 1) > (n_epoch - 30):
        lr -= (0.0002 / 30)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # scheduler.step()
    print(" lr:", scheduler.get_lr(), end='')
    if log_on:
        write.add_scalar("Learning Rate/1", scheduler.get_lr(), epoch)
        write.add_scalar("Learning Rate/2", optimizer.param_groups[0]['lr'], epoch)

        # Save the last train result
        save_image(predict.cpu().numpy().astype(np.uint8), log_file+'train')  # 保存最后一个训练（块）

        # Write train loss log
        write.add_scalar('Loss/train', total_loss / 150, epoch)
        write.add_scalar('Dice_train/CSF', dice[0] / 150, epoch)
        write.add_scalar('Dice_train/GM', dice[1] / 150, epoch)
        write.add_scalar('Dice_train/WM', dice[2] / 150, epoch)
        write.add_scalar('Dice_train/mean', dice.mean() / 150, epoch)

    # validation step-------------------------------------------------------
    torch.cuda.empty_cache()
    model.eval()  # 用了Batch Normalization就一定要区分训练和验证过程，因为验证和训练时BN的计算方式不同
    with torch.no_grad():  # Reduce memory consumption for computations with 'requires_grad=True'
        total_loss = 0
        i = 0
        for i in range(len(v_gt_list)):  # 有多个验证样本
            loss = 0
            out_list_p = []

            v_t1_list_p, v_t2_list_p, v_gt_list_p = \
                make_patch(v_t1_list, v_t2_list, v_gt_list, patch_size)

            j = 0
            for j in range(len(v_gt_list_p[i])):  # 每个验证样本 第i个验证样本的第j块
                t1_p = v_t1_list_p[i][j]
                t2_p = v_t2_list_p[i][j]
                gt_p = v_gt_list_p[i][j].cuda(gpu)

                if gt_p.sum() != 0:  # 非纯背景块
                    image = make_image(t1_p, t2_p).cuda(gpu)
                    # t1_p = t1_p.unsqueeze(dim=1).cuda(gpu)
                    # t2_p = t2_p.unsqueeze(dim=1).cuda(gpu)

                    out_p = model(image)  # out_p: 每一小块的分割结果
                    # out_p = model(t1_p, t2_p)  # out_p: 每一小块的分割结果

                    # 每块Loss
                    loss_p = 0
                    loss_p += lovasz_softmax(out_p, gt_p)
                    loss_p += Focal(output, gt_p, class_weight=[0.1, 0.4, 0.2, 0.3])
                    loss += loss_p  # 一个样本的总loss

                    # 保存每个块不能放在gpu里，会爆显存
                    out_list_p.append(out_p.cpu().detach().numpy())  # .detach()脱离梯度计算,不建议用.data
                else:  # 纯背景块如何处理？
                    out_list_p.append(torch.zeros_like(output).cpu().numpy())  # 直接压入全0块

            output = fuse(out_list_p, patch_size, [256, 192, 144]).cuda(gpu)  # output: 拼成完整图像的结果
            _, predict = torch.max(output, dim=1)  # 将预测概率值转变为预测结果: [0, 1, 2, 3]
            total_loss += loss / (j + 1)

            # 整体Dice
            v_gt = v_gt_list[i].cuda(gpu)
            v_gts = split_gt(v_gt)
            dice = get_dice(output, v_gts)
            if dice.mean() > best_dice.mean():
                best_dice = dice
                save_flag = True

            # Get the wrong mask
            wrong_mask = get_wrong(predict, v_gt.type(torch.long))

            print('\nval%d: loss:%.3f | csf:%.3f | gm:%.3f | wm:%.3f'
                  % (i, loss / (j + 1),
                     dice[0], dice[1], dice[2]),
                  end='')

        print()  # 换行

        if log_on:
            if save_flag:
                save_flag = False
                # Set the best validation result
                predic_image = (predict.cpu().numpy().astype(np.uint8), log_file+'valid'+str(i)+'-'+str(epoch))
                # save_image(predict.cpu().numpy().astype(np.uint8),
                #             log_file+'valid'+str(i)+'-'+str(epoch))  # 保存最后一个预测（图）
                # Set the wrong image
                wrong_image = (wrong_mask.cpu().numpy().astype(np.uint8), log_file+'wrong'+str(i)+'-'+str(epoch))
                # save_image(wrong_mask.cpu().numpy().astype(np.uint8),
                #             log_file+'wrong'+str(i)+'-'+str(epoch))  # 保存相应的错误图

            # Write validation loss log
            write.add_scalar('Loss/val', total_loss / (i + 1), epoch)
            write.add_scalar('Dice_val/CSF', dice[0] / (i + 1), epoch)
            write.add_scalar('Dice_val/GM', dice[1] / (i + 1), epoch)
            write.add_scalar('Dice_val/WM', dice[2] / (i + 1), epoch)
            write.add_scalar('Dice_val/mean', dice.mean() / (i+1), epoch)

        torch.cuda.empty_cache()

    if log_on:
        write.flush()

    # validation over ------- next epoch

if log_on:
    # Save the best validation result
    save_image(predic_image[0], predic_image[1])  # 保存最好的一个预测（图）
    # Save the wrong mask
    save_image(wrong_image[0], wrong_image[1])  # 保存相应的错误图

    write.close()
