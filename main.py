from network import *
from dataloader import *
import torch.optim as optim
from lovasz_losses import lovasz_softmax

data_path = '../datasets/iseg2017/'
patch_t1, patch_t2, patch_gt = load_train_data(data_path)

model = Unet(in_ch=2, out_ch=4).cuda(0)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for i in range(len(patch_t1)):
    t1 = patch_t1[i]
    t2 = patch_t2[i]
    gt = patch_gt[i].cuda(0)  # 0, 1, 2, 3

    t1 = t1.unsqueeze(dim=1)
    t2 = t2.unsqueeze(dim=1)
    # gt = gt.unsqueeze(dim=1)

    image = torch.cat((t1, t2), dim=1).to(device='cuda')
    # print(torch.cuda.max_memory_allocated())

    output = model(image)

    # todo: lovasz_softmax的原理，接收什么样的参数，是否需要对output或gt进行额外操作？
    loss = lovasz_softmax(output, gt)
    print(i, loss.item())

    loss.backward()  # 误差反向传播 : 计算梯度
    optimizer.step()  # 更新参数, 在梯度计算后使用
    optimizer.zero_grad()

    # todo: 添加Dice评价指标

    torch.cuda.empty_cache()
