import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import torch
from net import BEVNet
from database import ChileanDatasetOverlap
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.utils.tensorboard.writer import SummaryWriter
import loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(log_dir,
          coords_range_xyz=[-10., -10, -4, 10, 10, 8],
          div=[256, 256, 32],
          model="",
          test_mode=False):  # 新增参数
    """
    训练BEVNet网络

    Args:
        log_dir: 日志和模型保存目录
        coords_range_xyz: 空间范围 [xmin, ymin, zmin, xmax, ymax, zmax]
        div: 体素网格划分 [nx, ny, nz]
        model: 预训练模型路径（可选）
        test_mode: 是否为测试模式
    """
    writer = SummaryWriter()

    # 初始化网络
    net = BEVNet(div[2])
    net.to(device=device)
    print(net)

    # 根据模式选择配置
    if test_mode:
        # 测试模式：只用2个sessions，1000次迭代
        num_iter = 1000
        train_sessions = ['100', '101']
        print("*** 测试模式：只用2个sessions, 1000次迭代 ***")
    else:
        # 正式训练模式
        num_iter = 300050
        train_sessions = [str(i) for i in range(100, 160)]
        print(f"*** 正式训练：{len(train_sessions)}个sessions, {num_iter}次迭代 ***")

    # 创建Chilean数据集
    train_dataset = ChileanDatasetOverlap(
        sessions=train_sessions,
        root="/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/chilean_NoRot_NoScale/",
        pos_threshold_min=0,
        pos_threshold_max=7,
        neg_thresgold=35,
        coords_range_xyz=coords_range_xyz,
        div_n=div,
        random_rotation=True,
        random_occ=True,
        num_iter=num_iter)

    batch_size = 1  # 每个batch包含1对点云
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8)

    # 优化器配置
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        net.parameters()),
                                 lr=1e-4,
                                 weight_decay=1e-6)

    # 学习率调度器：每1000个batch衰减一次
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.99,
    )

    batch_num = 0

    # 加载预训练模型（如果提供）
    if not model == "":
        checkpoint = torch.load(model)
        net.load_state_dict(checkpoint['state_dict'], strict=True)
        # 可选：加载优化器状态
        # optimizer.load_state_dict(checkpoint['optimizer'])

    net.train()

    # 训练主循环
    for i_batch, sample_batch in tqdm(enumerate(train_loader),
                                      total=len(train_loader),
                                      desc='Train',
                                      leave=False):
        optimizer.zero_grad()

        # 拼接query和positive点云
        input = torch.cat([sample_batch['voxel0'], sample_batch['voxel1']],
                          dim=0).to(device)

        try:
            # 前向传播
            out, out4, x4 = net(input)

            # 分离两个点云的特征
            mask1 = (out.indices[:, 0] == 0)  # query点云的mask
            mask2 = (out.indices[:, 0] == 1)  # positive点云的mask

            # 计算配对损失（描述符、检测器、高度估计等）
            total_loss, desc_loss, det_loss, score_loss, z_loss, z_loss0, z_loss1, correct_ratio = loss.pair_loss(
                out.features[mask1, :],
                out.features[mask2, :],
                sample_batch['trans0'][0],
                sample_batch['trans1'][0],
                sample_batch['points0'][0],
                sample_batch['points1'][0],
                sample_batch['points_xy0'][0],
                sample_batch['points_xy1'][0],
                num_height=div[2],
                min_z=coords_range_xyz[2],
                height=coords_range_xyz[5] - coords_range_xyz[2],
                search_radiu=max(
                    (coords_range_xyz[3] - coords_range_xyz[0]) / div[0], 0.3))

            # 计算重叠区域损失
            loss4, precision4, recall4 = loss.overlap_loss(
                out4, sample_batch['trans0'][0], sample_batch['trans1'][0],
                coords_range_xyz[0], coords_range_xyz[3], False)

            # 计算粗特征损失
            desc_loss4, acc4 = loss.dist_loss(x4, sample_batch['trans0'][0],
                                              sample_batch['trans1'][0],
                                              coords_range_xyz[0],
                                              coords_range_xyz[3])

            # 组合总损失
            if not desc_loss4 is None:
                loss_all = desc_loss4 + loss4
            else:
                loss_all = loss4
            if not total_loss is None:
                loss_all = loss_all + total_loss

            # 反向传播
            loss_all.backward()
            optimizer.step()

            # 记录损失到TensorBoard
            with torch.no_grad():
                if not total_loss is None:
                    writer.add_scalar('desc loss',
                                      desc_loss.cpu().item(),
                                      global_step=batch_num)
                    writer.add_scalar('det loss',
                                      det_loss.cpu().item(),
                                      global_step=batch_num)
                    writer.add_scalar('score loss',
                                      score_loss.cpu().item(),
                                      global_step=batch_num)
                    writer.add_scalar('z loss',
                                      z_loss.cpu().item(),
                                      global_step=batch_num)
                    writer.add_scalar('z loss0',
                                      z_loss0.cpu().item(),
                                      global_step=batch_num)
                    writer.add_scalar('z loss1',
                                      z_loss1.cpu().item(),
                                      global_step=batch_num)
                    writer.add_scalar('correct ratio',
                                      correct_ratio,
                                      global_step=batch_num)
                writer.add_scalar('loss',
                                  loss_all.cpu().item(),
                                  global_step=batch_num)
                writer.add_scalar('loss4',
                                  loss4.cpu().item(),
                                  global_step=batch_num)
                if not desc_loss4 is None:
                    writer.add_scalar('desc loss4',
                                      desc_loss4.cpu().item(),
                                      global_step=batch_num)
                    writer.add_scalar('acc4',
                                      acc4.cpu().item(),
                                      global_step=batch_num)
                writer.add_scalar('precision4',
                                  precision4,
                                  global_step=batch_num)
                writer.add_scalar('recall4', recall4, global_step=batch_num)
                writer.add_scalar(
                    'LR',
                    optimizer.state_dict()['param_groups'][0]['lr'],
                    global_step=batch_num)

                batch_num += 1

                # 每1000个batch保存一次模型
                if batch_num % 1000 == 10:
                    torch.save(
                        {
                            'state_dict': net.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'batch_num': batch_num
                        }, os.path.join(log_dir,
                                        str(batch_num) + '.ckpt'))

            # 每1000个batch更新学习率
            if batch_num % 1000 == 0 and batch_num:
                scheduler.step()

        except Exception as inst:
            print(inst)


if __name__ == '__main__':
    # 日志目录
    log_dir = './log_chilean'
    if (not os.path.exists(log_dir)):
        os.makedirs(log_dir)

    # 开始训练
    # train(log_dir,
    #       coords_range_xyz=[-10., -10, -4, 10, 10, 8],  # Chilean数据集空间范围
    #       div=[256, 256, 32],  # 体素网格划分
    #       model="")  # 预训练模型路径（空表示从头训练）
    train(log_dir,
          coords_range_xyz=[-10., -10, -4, 10, 10, 8],
          div=[256, 256, 32],
          model="",
          test_mode=True)  # 开启测试模式