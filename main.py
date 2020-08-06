from network import *
from dataloader import *
from evaluation import *
from utility import *
import torch.optim as optim
from lovasz_losses import lovasz_softmax
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('gpu_id', nargs='?', default=3, type=int, help='Assign gpu id')
args = parser.parse_args()

gpu = args.gpu_id
cuda = torch.device(gpu)

val_idx = 0  # validation index
data_path = '../dataset/iseg2017/'
t1_list, t2_list, gt_list, v_t1_list, v_t2_list, v_gt_list = \
    load_data(data_path, val_idx)  # train & validation set

model = Unet(in_ch=2, out_ch=4).cuda(gpu)
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters())

iterations = 150  # 一个epoch中的迭代次数
for epoch in range(100):
    print('epoch', epoch + 1)
    total_loss = 0
    dice = np.zeros(3)
    patch_size = 64

    # train step--------------------------------------------------------
    for i in range(iterations):
        t1, t2, gt = random_patch(t1_list, t2_list, gt_list, patch_size)

        image = make_image(t1, t2).cuda(gpu)
        gt = gt.cuda(gpu)

        output = model(image)

        # Dice
        gts = split_gt(gt)
        dice += get_dice(output, gts)

        loss = lovasz_softmax(output, gt)
        total_loss += loss.item()

        # 打印结果
        progress_bar(i, iterations)
        print(' %3d/%3d | loss:%.3f | csf:%.3f | gm:%.3f | wm:%.3f'
              % (i + 1, iterations,
                 total_loss / (i + 1),
                 dice[0] / (i + 1),
                 dice[1] / (i + 1),
                 dice[2] / (i + 1)),
              end='')

        loss.backward()  # Error Backpropagation for computing gradients
        optimizer.step()  # Update parameters, base on the gradients
        optimizer.zero_grad()

    # validation step-------------------------------------------------------
    with torch.no_grad():  # Reduce memory consumption for computations with 'requires_grad=True'
        print("\nValidate:", end='')
        for i in range(len(v_t1_list)):
            output_list = []

            for j in range(len(v_t1_list[i])):
                t1 = v_t1_list[i][j]
                t2 = v_t2_list[i][j]

                if t1.sum() != 0:
                    t1 = t1.unsqueeze(dim=1)  # 增加一个维度
                    t2 = t2.unsqueeze(dim=1)

                    image = torch.cat((t1, t2), dim=1).cuda(gpu)

                    output = model(image)
                    output_list.append(output.cpu().detach().numpy())  # .detach()脱离梯度计算,不建议用.data
                else:
                    output_list.append(torch.zeros_like(output).cpu().numpy())

            prediction = fuse(output_list, patch_size, v_gt_list[i].shape).cuda(gpu)
            gt = v_gt_list[i].cuda(gpu)
            loss = lovasz_softmax(prediction, gt)

            # Dice
            dice = np.zeros(3)
            gts = split_gt(gt)
            dice += get_dice(prediction, gts)

            print(' loss:%.3f | csf:%.3f | gm:%.3f | wm:%.3f'
                  % (loss / (i + 1),
                     dice[0] / (i + 1),
                     dice[1] / (i + 1),
                     dice[2] / (i + 1)),
                  end='')

        print()  # 换行
