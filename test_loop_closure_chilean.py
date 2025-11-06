from net import BEVNet
import torch
import numpy as np
from tqdm import tqdm
import os
import open3d as o3d
import utils
import math
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_feature_chilean(model, files, batch_num=512):
    """
    批量提取Chilean点云的BEV特征

    Args:
        model: BEVNet模型
        files: 点云文件路径列表
        batch_num: 批处理大小
    """
    for q_index in tqdm(range(len(files) // batch_num),
                        total=len(files) // batch_num):
        batch_files = files[q_index * batch_num:(q_index + 1) * batch_num]

        # 加载点云并转换（float64→float32）
        queries = utils.load_pc_files(batch_files,
                                      coords_range_xyz=[-10., -10, -4, 10, 10, 8],
                                      div_n=[256, 256, 32],
                                      dtype_in='float64')

        with torch.no_grad():
            feed_tensor = torch.tensor(queries).float()
            feed_tensor = feed_tensor.to(next(model.parameters()).device)
            q_out = model.extract_feature(feed_tensor)
            q_out = q_out.detach().cpu().numpy()

            # 保存特征文件
            for i in range(len(batch_files)):
                file = batch_files[i].replace('pointcloud_20m_10overlap', "BEV_FEATURE").replace(
                    '.bin', '.npy')
                # 确保目标目录存在
                os.makedirs(os.path.dirname(file), exist_ok=True)
                np.save(file, q_out[i])

    # 处理剩余的文件
    index_edge = len(files) // batch_num * batch_num
    if index_edge < len(files):
        batch_files = files[index_edge:len(files)]
        queries = utils.load_pc_files(batch_files,
                                      coords_range_xyz=[-10., -10, -4, 10, 10, 8],
                                      div_n=[256, 256, 32],
                                      dtype_in='float64')
        with torch.no_grad():
            feed_tensor = torch.tensor(queries).float()
            feed_tensor = feed_tensor.to(next(model.parameters()).device)
            q_out = model.extract_feature(feed_tensor).detach().cpu().numpy()
            for i in range(len(batch_files)):
                file = batch_files[i].replace('pointcloud_20m_10overlap', "BEV_FEATURE").replace(
                    '.bin', '.npy')
                os.makedirs(os.path.dirname(file), exist_ok=True)
                np.save(file, q_out[i])


def extract_chilean(model,
                    session,
                    batch_num=32,
                    root="/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/chilean_NoRot_NoScale/"):
    """
    提取Chilean单个session的特征

    Args:
        model: BEVNet模型
        session: session ID（如'160'）
        batch_num: 批处理大小
        root: 数据集根目录
    """
    model.train()

    # 点云文件夹
    folder = os.path.join(root, session, "pointcloud_20m_10overlap")
    # 特征输出文件夹
    out_folder = os.path.join(root, session, "BEV_FEATURE")
    if (not os.path.exists(out_folder)):
        os.makedirs(out_folder)

    # 获取所有点云文件
    pcd_files = os.listdir(folder)
    pcd_files.sort()
    pcd_files = [os.path.join(folder, v) for v in pcd_files]

    # 读取位姿信息（用于后续评估）
    csv_path = os.path.join(root, session, 'pointcloud_pos_ori_20m_10overlap.csv')
    df = pd.read_csv(csv_path, sep=',')
    pose_xy = df[['x', 'y']].values  # 提取XY坐标

    print(f"Extracting features for session {session}: {len(pcd_files)} files")
    extract_feature_chilean(model, pcd_files, batch_num)


def evaluate_chilean(model,
                     session="160",
                     batch_num=1,
                     th_min=0,
                     th_max=10,
                     th_max_pre=20,
                     skip=50,
                     root="/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/chilean_NoRot_NoScale/"):
    """
    评估Chilean数据集的loop closure性能

    Args:
        model: BEVNet模型
        session: 测试session ID
        batch_num: 批处理大小
        th_min: 最小距离阈值（排除过近的点）
        th_max: 查询的最大距离阈值
        th_max_pre: 预测正确的最大距离阈值
        skip: 跳过最近的N帧
        root: 数据集根目录
    """
    model.train()

    # 特征文件夹
    folder = os.path.join(root, session, 'BEV_FEATURE')
    feature_files = os.listdir(folder)
    feature_files.sort()
    valid_pose_id = [int(v.split('.')[0]) for v in feature_files]
    feature_files = [os.path.join(folder, v) for v in feature_files]

    # 读取位姿
    csv_path = os.path.join(root, session, 'pointcloud_pos_ori_20m_10overlap.csv')
    df = pd.read_csv(csv_path, sep=',')
    pose_xy = df[['x', 'y']].values[valid_pose_id]  # 只取有效的位姿

    pos_num = 0
    neg_num = 0

    # 遍历每个query点云
    for i in tqdm(range(len(feature_files)), total=len(feature_files)):
        # 计算与所有点的距离
        diff = pose_xy - pose_xy[i]
        dis = np.linalg.norm(diff, axis=1)
        max_d = 100000

        # 排除最近的skip帧
        dis[max(i - skip, 0):] = max_d

        # 排除过近的点
        mask = (dis < th_min)
        dis[mask] = max_d

        minid_gt = np.argmin(dis)
        temp_min = np.min(dis)

        # 如果存在足够近的真值点
        if temp_min < th_max:
            # 计算overlap分数
            scores = np.zeros(len(feature_files), dtype='float32')
            feai = utils.load_npy_files([feature_files[i]])
            feai = feai.repeat(batch_num, axis=0)
            feai = torch.from_numpy(feai).to(next(model.parameters()).device)

            st = 0
            # 批量计算overlap
            while st < min(len(feature_files), i - skip):
                ed = min(st + batch_num, len(feature_files))
                batch_files = feature_files[st:ed]
                feai = feai[0:len(batch_files)]
                feaj = utils.load_npy_files(batch_files)
                feaj = torch.from_numpy(feaj).to(
                    next(model.parameters()).device)

                # 计算overlap分数
                overlap = model.calc_overlap(
                    torch.cat([feai, feaj]).permute(0, 2, 3, 1))
                scores[st:ed] = overlap.detach().cpu().numpy()
                st = ed

            # 排除不符合条件的候选
            scores[max(i - skip, 0):] = 0
            scores[mask] = 0

            # 找到分数最高的匹配
            minid = np.argmax(scores)
            mindis = math.sqrt((pose_xy[i, 0] - pose_xy[minid, 0]) ** 2 +
                               (pose_xy[i, 1] - pose_xy[minid, 1]) ** 2)

            # 判断是否正确
            if mindis < th_max_pre:
                pos_num += 1
            else:
                neg_num += 1

            print("recall:", pos_num / (pos_num + neg_num), mindis, temp_min,
                  pos_num, neg_num + pos_num, i, minid, minid_gt)


if __name__ == "__main__":
    # 加载模型
    net = BEVNet(32).to(device=device)
    checkpoint = torch.load('./log_chilean/best_model.ckpt')  # 使用训练好的模型
    net.load_state_dict(checkpoint['state_dict'], strict=True)
    net.train()

    # 测试sessions: 160-209
    test_sessions = [str(i) for i in range(160, 210)]

    # 选择一个或多个session进行测试
    test_session = "160"  # 可以改为其他session

    print(f"Extracting features for session {test_session}...")
    extract_chilean(net, test_session, batch_num=64)

    print(f"Evaluating session {test_session}...")
    evaluate_chilean(net,
                     session=test_session,
                     batch_num=1,
                     th_min=0,
                     th_max=5,
                     th_max_pre=10,
                     skip=50)

    # 如果要测试所有测试sessions，可以使用循环：
    # for session in test_sessions:
    #     print(f"\n{'='*60}")
    #     print(f"Processing session {session}")
    #     print(f"{'='*60}")
    #     extract_chilean(net, session, batch_num=64)
    #     evaluate_chilean(net, session=session, batch_num=1,
    #                     th_min=0, th_max=5, th_max_pre=10, skip=50)