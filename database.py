from typing_extensions import Self
from sklearn import datasets
from torch.utils.data import Dataset
import torch
import os
import sys

# 修改import路径以适配Chilean数据集
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
import numpy as np
from matplotlib import pyplot as plt
import random
from scipy.spatial.transform import Rotation as R
import glob
import pathlib
import open3d as o3d
import copy
from scipy.linalg import expm, norm
from spconv.pytorch.utils import PointToVoxel
from sklearn.neighbors import KDTree
from tqdm import tqdm
import torch_scatter
import time
import pandas as pd


class ChileanDatasetOverlap(Dataset):
    """
    Chilean地下矿井数据集，用于BEVNet训练

    数据结构：
    - 多个session文件夹(100-209)
    - 每个session包含：pointcloud_20m_10overlap/*.bin 和 pointcloud_pos_ori_20m_10overlap.csv
    - CSV格式：timestamp, x, y, z, qx, qy, qz, qw
    """

    def __init__(self,
                 sessions=['100', '101', '102'],  # session列表，如['100', '101', ...]
                 root="/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/chilean_NoRot_NoScale/",
                 pos_threshold_min=0,
                 pos_threshold_max=7,
                 neg_thresgold=35,
                 coords_range_xyz=[-10., -10, -4, 10, 10, 8],
                 div_n=[256, 256, 32],
                 random_rotation=True,
                 random_occ=False,
                 num_iter=300000) -> None:
        super().__init__()
        self.num_iter = num_iter
        self.icp_path = os.path.join(root, "icp_chilean")  # ICP缓存路径
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)
        self.div_n = div_n
        self.coords_range_xyz = coords_range_xyz
        self.chilean_icp_cache = {}
        self.random_rotation = random_rotation
        self.random_occ = random_occ
        self.device = torch.device('cpu')
        self.randg = np.random.RandomState()
        self.root = root
        self.sessions = sessions
        self.poses = []  # 存储所有位姿

        # 遍历所有session，读取位姿信息
        for session in sessions:
            csv_path = os.path.join(root, session, 'pointcloud_pos_ori_20m_10overlap.csv')
            if not os.path.exists(csv_path):
                print(f"Warning: {csv_path} not found, skipping session {session}")
                continue

            # 读取CSV (tab分隔)
            df = pd.read_csv(csv_path, sep=',')

            # 构建4x4变换矩阵列表
            session_poses = []
            for _, row in df.iterrows():
                T = utils.quat_xyzw_to_SE3(
                    row['x'], row['y'], row['z'],
                    row['qx'], row['qy'], row['qz'], row['qw']
                )
                session_poses.append(T)

            self.poses.append(np.array(session_poses))

        # 构建正负样本对
        key = 0
        acc_num = 0
        self.pairs = {}

        for i in range(len(self.poses)):
            # 提取XY坐标用于距离计算
            pose_xy = self.poses[i][:, :2, 3]  # (N, 2) 取变换矩阵的XY平移部分

            # 计算距离矩阵
            inner = 2 * np.matmul(pose_xy, pose_xy.T)
            xx = np.sum(pose_xy ** 2, 1, keepdims=True)
            dis = xx - inner + xx.T
            dis = np.sqrt(np.abs(dis))

            # 正样本：pos_threshold_min < 距离 < pos_threshold_max
            id_pos = np.argwhere((dis < pos_threshold_max) & (dis > pos_threshold_min))
            # 负样本：距离 < neg_threshold的都算相关区域
            id_neg = np.argwhere(dis < neg_thresgold)

            for j in range(len(pose_xy)):
                positives = id_pos[:, 1][id_pos[:, 0] == j] + acc_num
                negatives = id_neg[:, 1][id_neg[:, 0] == j] + acc_num
                self.pairs[key] = {
                    "query_session": i,
                    "query_id": j,
                    "positives": positives.tolist(),
                    "negatives": set(negatives.tolist())
                }
                key += 1
            acc_num += len(pose_xy)

        self.all_ids = set(range(len(self.pairs)))
        print(f"ChileanDataset initialized: {len(self.sessions)} sessions, {len(self.pairs)} pairs")

    def get_random_positive(self, idx):
        """随机获取一个正样本"""
        positives = self.pairs[idx]["positives"]
        if len(positives) == 0:
            return idx  # 如果没有正样本，返回自己
        randid = random.randint(0, len(positives) - 1)
        return positives[randid]

    def get_random_negative(self, idx):
        """随机获取一个负样本"""
        negatives = list(self.all_ids - self.pairs[idx]["negatives"])
        if len(negatives) == 0:
            return idx
        randid = random.randint(0, len(negatives) - 1)
        return negatives[randid]

    def load_pcd(self, idx):
        """
        加载点云数据

        Args:
            idx: 全局索引

        Returns:
            点云数据 (N, 3)
        """
        query = self.pairs[idx]
        session = self.sessions[query["query_session"]]
        local_id = query["query_id"]

        # 从CSV中获取timestamp
        csv_path = os.path.join(self.root, session, 'pointcloud_pos_ori_20m_10overlap.csv')
        df = pd.read_csv(csv_path, sep=',')
        timestamp = str(int(df.iloc[local_id]['timestamp']))

        # 构建bin文件路径
        file = os.path.join(self.root, session, "pointcloud_20m_10overlap", timestamp + '.bin')

        # 加载float64并转换为float32
        return np.fromfile(file, dtype='float64').reshape(-1, 3).astype('float32')

    def get_icp_name(self, query_id, pos_id):
        """生成ICP缓存文件名"""
        query = self.pairs[query_id]
        pos = self.pairs[pos_id]
        session_q = self.sessions[query["query_session"]]
        session_p = self.sessions[pos["query_session"]]
        t0 = query["query_id"]
        t1 = pos["query_id"]
        key = f'{session_q}_{t0}_{session_p}_{t1}'
        return os.path.join(self.icp_path, key + '.npy')

    def get_odometry(self, idx):
        """获取位姿变换矩阵"""
        query = self.pairs[idx]
        session_idx = query["query_session"]
        local_id = query["query_id"]
        return self.poses[session_idx][local_id]

    def __len__(self):
        return self.num_iter

    def __getitem__(self, idx):
        """
        获取训练样本

        Returns:
            dict: {
                'voxel0': 体素化的query点云,
                'voxel1': 体素化的positive点云,
                'trans0': query的变换矩阵,
                'trans1': positive的变换矩阵,
                'time': 数据加载时间,
                'points0': query原始点,
                'points1': positive原始点,
                'points_xy0': query的XY投影点,
                'points_xy1': positive的XY投影点
            }
        """
        time0 = time.time()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        queryid = idx % len(self.pairs)
        posid = self.get_random_positive(queryid)

        query_points = self.load_pcd(queryid)
        pos_points = self.load_pcd(posid)

        query_odom = self.get_odometry(queryid)
        pos_odom = self.get_odometry(posid)

        # 计算相对变换或使用ICP
        filename = self.get_icp_name(queryid, posid)
        if filename not in self.chilean_icp_cache:
            if not os.path.exists(filename):
                # 计算初始相对变换
                M = query_odom.T @ np.linalg.inv(pos_odom.T)
                M = M.T

                # 使用ICP精细对齐
                query_points_t = utils.apply_transform(query_points, M)
                pcd0 = utils.make_open3d_point_cloud(query_points_t)
                pcd1 = utils.make_open3d_point_cloud(pos_points)
                reg = o3d.pipelines.registration.registration_icp(
                    pcd0, pcd1, 0.2, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
                M2 = M @ reg.transformation
                np.save(filename, M2)
            else:
                try:
                    M2 = np.load(filename)
                except Exception as inst:
                    print(inst)
                    M = query_odom.T @ np.linalg.inv(pos_odom.T)
                    M = M.T
                    query_points_t = utils.apply_transform(query_points, M)
                    pcd0 = utils.make_open3d_point_cloud(query_points_t)
                    pcd1 = utils.make_open3d_point_cloud(pos_points)
                    reg = o3d.pipelines.registration.registration_icp(
                        pcd0, pcd1, 0.2, np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
                    M2 = M @ reg.transformation
                    np.save(filename, M2)
            self.chilean_icp_cache[filename] = M2
        else:
            M2 = self.chilean_icp_cache[filename]

        # 随机旋转数据增强
        if self.random_rotation:
            T0 = utils.rot3d(2, 2. * self.randg.rand(1) * np.pi)
            T1 = utils.rot3d(2, 2. * self.randg.rand(1) * np.pi)
            trans = T1 @ M2 @ np.linalg.inv(T0)
            query_points = utils.apply_transform(query_points, T0)
            pos_points = utils.apply_transform(pos_points, T1)
        else:
            trans = M2

        # 随机遮挡数据增强
        if self.random_occ:
            query_points = utils.occ_pcd(query_points, state_st=6, max_range=np.pi)
            pos_points = utils.occ_pcd(pos_points, state_st=6, max_range=np.pi)

        # 体素化
        ids0, points0, ids_xy0, points_xy0 = utils.load_voxel(
            query_points, self.coords_range_xyz, self.div_n)
        ids1, points1, ids_xy1, points_xy1 = utils.load_voxel(
            pos_points, self.coords_range_xyz, self.div_n)

        voxel_out0 = np.zeros(self.div_n, dtype='float32')
        voxel_out0[ids0[:, 0], ids0[:, 1], ids0[:, 2]] = 1
        voxel_out1 = np.zeros(self.div_n, dtype='float32')
        voxel_out1[ids1[:, 0], ids1[:, 1], ids1[:, 2]] = 1

        time1 = time.time()
        return {
            "voxel0": voxel_out0,
            "voxel1": voxel_out1,
            "trans0": trans.astype('float32'),
            "trans1": np.identity(4, dtype='float32'),
            "time": time1 - time0,
            "points0": points0,
            "points1": points1,
            "points_xy0": points_xy0,
            "points_xy1": points_xy1
        }


class KITTIDatasetOverlap(Dataset):
    """KITTI数据集（保持原有实现）"""

    def __init__(self,
                 sequs=[
                     '00', '01', '02', '03', '04', '05', '06', '07', '08',
                     '09', '10'
                 ],
                 root="/media/l/yp2/KITTI/odometry/dataset/sequences/",
                 pos_threshold_min=10,
                 pos_threshold_max=20,
                 neg_thresgold=50,
                 coords_range_xyz=[-50., -50, -4, 50, 50, 3],
                 div_n=[256, 256, 32],
                 random_rotation=True,
                 random_occ=False,
                 num_iter=300000) -> None:
        super().__init__()
        self.num_iter = num_iter
        self.icp_path = root.replace("sequences", "icp")
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)
        self.div_n = div_n
        self.coords_range_xyz = coords_range_xyz
        self.kitti_icp_cache = {}
        self.random_rotation = random_rotation
        self.random_occ = random_occ
        self.device = torch.device('cpu')
        self.randg = np.random.RandomState()
        self.root = root
        self.sequs = sequs
        self.poses = []
        for seq in sequs:
            pose = np.genfromtxt(os.path.join(root, seq, 'poses.txt'))
            self.poses.append(pose)
        key = 0
        acc_num = 0
        self.pairs = {}
        for i in range(len(self.poses)):
            pose = self.poses[i][:, [3, 11]]
            inner = 2 * np.matmul(pose, pose.T)
            xx = np.sum(pose ** 2, 1, keepdims=True)
            dis = xx - inner + xx.T
            dis = np.sqrt(np.abs(dis))
            id_pos = np.argwhere((dis < pos_threshold_max)
                                 & (dis > pos_threshold_min))
            id_neg = np.argwhere(dis < neg_thresgold)
            for j in range(len(pose)):
                positives = id_pos[:, 1][id_pos[:, 0] == j] + acc_num
                negatives = id_neg[:, 1][id_neg[:, 0] == j] + acc_num
                self.pairs[key] = {
                    "query_seq": i,
                    "query_id": j,
                    "positives": positives.tolist(),
                    "negatives": set(negatives.tolist())
                }
                key += 1
            acc_num += len(pose)
        self.all_ids = set(range(len(self.pairs)))

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02,
                7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03,
                1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02,
                          -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_random_positive(self, idx):
        positives = self.pairs[idx]["positives"]
        randid = random.randint(0, len(positives) - 1)
        return positives[randid]

    def get_random_negative(self, idx):
        negatives = list(self.all_ids - self.pairs[idx]["negatives"])
        randid = random.randint(0, len(negatives) - 1)
        return negatives[randid]

    def load_pcd(self, idx):
        query = self.pairs[idx]
        seq = self.sequs[query["query_seq"]]
        id = str(query["query_id"]).zfill(6)
        file = os.path.join(self.root, seq, "velodyne", id + '.bin')
        return np.fromfile(file, dtype='float32').reshape(-1, 4)[:, 0:3]

    def get_icp_name(self, query_id, pos_id):
        query = self.pairs[query_id]
        drive = int(self.sequs[query["query_seq"]])
        t0 = query["query_id"]
        pos = self.pairs[pos_id]
        t1 = pos["query_id"]
        key = '%d_%d_%d' % (drive, t0, t1)
        return os.path.join(self.icp_path, key + '.npy')

    def get_odometry(self, idx):
        query = self.pairs[idx]
        T_w_cam0 = self.poses[query["query_seq"]][query["query_id"]].reshape(
            3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0

    def __len__(self):
        return self.num_iter

    def __getitem__(self, idx):
        time0 = time.time()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        queryid = idx % len(self.pairs)
        posid = self.get_random_positive(queryid)
        query_points = self.load_pcd(queryid)
        pos_points = self.load_pcd(posid)
        query_odom = self.get_odometry(queryid)
        pos_odom = self.get_odometry(posid)
        filename = self.get_icp_name(queryid, posid)
        if filename not in self.kitti_icp_cache:
            if not os.path.exists(filename):
                M = (self.velo2cam @ query_odom.T @ np.linalg.inv(pos_odom.T)
                     @ np.linalg.inv(self.velo2cam)).T
                query_points_t = utils.apply_transform(query_points, M)
                pcd0 = utils.make_open3d_point_cloud(query_points_t)
                pcd1 = utils.make_open3d_point_cloud(pos_points)
                reg = o3d.pipelines.registration.registration_icp(
                    pcd0, pcd1, 0.2, np.eye(4),
                    o3d.pipelines.registration.
                    TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=200))
                M2 = M @ reg.transformation
                np.save(filename, M2)
            else:
                try:
                    M2 = np.load(filename)
                except Exception as inst:
                    print(inst)
                    M = (self.velo2cam @ query_odom.T @ np.linalg.inv(
                        pos_odom.T) @ np.linalg.inv(self.velo2cam)).T
                    query_points_t = utils.apply_transform(query_points, M)
                    pcd0 = utils.make_open3d_point_cloud(query_points_t)
                    pcd1 = utils.make_open3d_point_cloud(pos_points)
                    reg = o3d.pipelines.registration.registration_icp(
                        pcd0, pcd1, 0.2, np.eye(4),
                        o3d.pipelines.registration.
                        TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(
                            max_iteration=200))
                    M2 = M @ reg.transformation
                    np.save(filename, M2)
            self.kitti_icp_cache[filename] = M2
        else:
            M2 = self.kitti_icp_cache[filename]

        if self.random_rotation:
            T0 = utils.rot3d(2, 2. * self.randg.rand(1) * np.pi)
            T1 = utils.rot3d(2, 2. * self.randg.rand(1) * np.pi)
            trans = T1 @ M2 @ np.linalg.inv(T0)
            query_points = utils.apply_transform(query_points, T0)
            pos_points = utils.apply_transform(pos_points, T1)
        else:
            trans = M2
        if self.random_occ:
            query_points = utils.occ_pcd(query_points,
                                         state_st=6,
                                         max_range=np.pi)
            pos_points = utils.occ_pcd(pos_points, state_st=6, max_range=np.pi)

        ids0, points0, ids_xy0, points_xy0 = utils.load_voxel(
            query_points, self.coords_range_xyz, self.div_n)
        ids1, points1, ids_xy1, points_xy1 = utils.load_voxel(
            pos_points, self.coords_range_xyz, self.div_n)

        voxel_out0 = np.zeros(self.div_n, dtype='float32')
        voxel_out0[ids0[:, 0], ids0[:, 1], ids0[:, 2]] = 1
        voxel_out1 = np.zeros(self.div_n, dtype='float32')
        voxel_out1[ids1[:, 0], ids1[:, 1], ids1[:, 2]] = 1
        time1 = time.time()
        return {
            "voxel0": voxel_out0,
            "voxel1": voxel_out1,
            "trans0": trans.astype('float32'),
            "trans1": np.identity(4, dtype='float32'),
            "time": time1 - time0,
            "points0": points0,
            "points1": points1,
            "points_xy0": points_xy0,
            "points_xy1": points_xy1
        }


class GuangzhouDatasetOverlap(Dataset):
    """Guangzhou数据集（保持原有实现）"""

    def __init__(self,
                 root='/media/l/yp2/BAIDU/guangzhou',
                 sequ="MKZ103_20220119095732",
                 pos_min=30,
                 pos_max=70,
                 coords_range_xyz=[-50., -50, -3, 50, 50, 7],
                 div_n=[256, 256, 32],
                 random_rotation=True,
                 num_iter=30000) -> None:
        super().__init__()
        self.num_iter = num_iter
        self.root = root
        self.sequ = sequ
        self.coords_range_xyz = coords_range_xyz
        self.div_n = div_n
        self.randg = np.random.RandomState()
        self.random_rotation = random_rotation
        pose_file = os.path.join(root, sequ, 'pose.txt')
        self.stamps, self.poses = utils.read_poses(pose_file, dx=2, st=100)
        self.poses = np.array([utils.se3_to_SE3(p) for p in self.poses])
        pose_xy = self.poses[:, 0:2, 3]
        self.pairs = {}
        for i in tqdm(range(len(pose_xy)),
                      total=len(pose_xy),
                      desc="Gen pairs...."):
            posei = np.zeros_like(pose_xy)
            posei[:, 0] = pose_xy[i, 0]
            posei[:, 1] = pose_xy[i, 1]
            diff = pose_xy - posei
            dist = np.linalg.norm(diff, axis=1)
            self.pairs[i] = {
                "query":
                    i,
                "positives":
                    np.argwhere((dist > pos_min) & (dist < pos_max)).reshape(-1)
            }

    def __len__(self):
        return self.num_iter

    def __getitem__(self, idx):
        time0 = time.time()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        queryid = idx % len(self.pairs)
        posid = self.get_random_positive(queryid)
        pcd_file_query = os.path.join(self.root, self.sequ, "submap",
                                      str(self.stamps[queryid]) + ".pcd")
        pcd_file_pos = os.path.join(self.root, self.sequ, "submap",
                                    str(self.stamps[posid]) + ".pcd")
        points_query = utils.load_pcd(pcd_file_query)
        points_pos = utils.load_pcd(pcd_file_pos)
        trans_query = self.poses[queryid]
        trans_pos = self.poses[posid]

        if self.random_rotation:
            T0 = utils.rot3d(2, 2. * self.randg.rand(1) * np.pi)
            T1 = utils.rot3d(2, 2. * self.randg.rand(1) * np.pi)
            trans_query = trans_query @ np.linalg.inv(T0)
            trans_pos = trans_pos @ np.linalg.inv(T1)
            points_query = utils.apply_transform(points_query, T0)
            points_pos = utils.apply_transform(points_pos, T1)

        ids0, points0, ids_xy0, points_xy0 = utils.load_voxel(
            points_query, self.coords_range_xyz, self.div_n)
        ids1, points1, ids_xy1, points_xy1 = utils.load_voxel(
            points_pos, self.coords_range_xyz, self.div_n)

        voxel_out0 = np.zeros(self.div_n, dtype='float32')
        voxel_out0[ids0[:, 0], ids0[:, 1], ids0[:, 2]] = 1
        voxel_out1 = np.zeros(self.div_n, dtype='float32')
        voxel_out1[ids1[:, 0], ids1[:, 1], ids1[:, 2]] = 1
        time1 = time.time()
        return {
            "voxel0": voxel_out0,
            "voxel1": voxel_out1,
            "trans0": trans_query.astype('float32'),
            "trans1": trans_pos.astype('float32'),
            "time": time1 - time0,
            "points0": points0,
            "points1": points1,
            "points_xy0": points_xy0,
            "points_xy1": points_xy1
        }

    def get_random_positive(self, idx):
        positives = self.pairs[idx]["positives"]
        randid = random.randint(0, len(positives) - 1)
        return positives[randid]


class KITTIPairDataset(torch.utils.data.Dataset):
    """KITTI配对数据集（保持原有实现）"""

    def __init__(self,
                 subset_names=[8, 9, 10],
                 root='/media/l/yp2/KITTI/odometry/dataset/',
                 random_rotation=True,
                 coords_range_xyz=[-50., -50, -4, 50, 50, 3],
                 div=[256, 256, 32]):
        self.files = []
        self.kitti_cache = {}
        self.coords_range_xyz = coords_range_xyz
        self.div = div
        self.matching_search_voxel_size = 0.3

        self.random_rotation = random_rotation
        self.randg = np.random.RandomState()
        self.root = root
        self.kitti_icp_cache = {}

        self.icp_path = os.path.join(root, "icp")
        pathlib.Path(self.icp_path).mkdir(parents=True, exist_ok=True)

        for dirname in subset_names:
            drive_id = int(dirname)
            inames = self.get_all_scan_ids(drive_id)
            all_odo = self.get_video_odometry(drive_id, return_all=True)
            all_pos = np.array(
                [self.odometry_to_positions(odo) for odo in all_odo])
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1))
            more_than_10 = pdist > 10
            curr_time = inames[0]
            while curr_time in inames:
                next_time = np.where(
                    more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    self.files.append((drive_id, curr_time, next_time))
                    curr_time = next_time + 1
        if (8, 15, 58) in self.files:
            self.files.remove((8, 15, 58))

    def __len__(self):
        return len(self.files)

    def get_all_scan_ids(self, drive_id):
        fnames = glob.glob(self.root +
                           '/sequences/%02d/velodyne/*.bin' % drive_id)

        assert len(
            fnames
        ) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        inames.sort()
        return inames

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02,
                7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03,
                1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02,
                          -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self,
                           drive,
                           indices=None,
                           ext='.txt',
                           return_all=False):
        data_path = self.root + '/poses_raw/%02d.txt' % drive
        if data_path not in self.kitti_cache:
            self.kitti_cache[data_path] = np.genfromtxt(data_path)
        if return_all:
            return self.kitti_cache[data_path]
        else:
            return self.kitti_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        T_w_cam0 = odometry.reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0

    def get_matching_indices(self,
                             source,
                             target,
                             trans,
                             search_voxel_size,
                             K=None):
        source_copy = copy.deepcopy(source)
        target_copy = copy.deepcopy(target)
        source_copy.transform(trans)
        pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

        match_inds = []
        for i, point in enumerate(source_copy.points):
            [_, idx,
             _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
            if K is not None:
                idx = idx[:K]
            for j in idx:
                match_inds.append((i, j))
        return match_inds

    def _get_velodyne_fn(self, drive, t):
        fname = os.path.join(self.root,
                             'sequences/%02d/velodyne/%06d.bin' % (drive, t))
        return fname

    def __getitem__(self, idx):
        time0 = time.time()
        drive = self.files[idx][0]
        t0, t1 = self.files[idx][1], self.files[idx][2]
        all_odometry = self.get_video_odometry(drive, [t0, t1])
        positions = [
            self.odometry_to_positions(odometry) for odometry in all_odometry
        ]
        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        xyz0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)[:, :3]
        xyz1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)[:, :3]

        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in self.kitti_icp_cache:
            if not os.path.exists(filename):
                M = (self.velo2cam @ positions[0].T @ np.linalg.inv(
                    positions[1].T) @ np.linalg.inv(self.velo2cam)).T
                xyz0_t = utils.apply_transform(xyz0, M)
                pcd0 = utils.make_open3d_point_cloud(xyz0_t)
                pcd1 = utils.make_open3d_point_cloud(xyz1)
                reg = o3d.pipelines.registration.registration_icp(
                    pcd0, pcd1, 0.2, np.eye(4),
                    o3d.pipelines.registration.
                    TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=200))
                M2 = M @ reg.transformation
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            self.kitti_icp_cache[key] = M2
        else:
            M2 = self.kitti_icp_cache[key]

        if self.random_rotation:
            T0 = utils.rot3d(2, 2. * self.randg.rand(1) * np.pi)
            T1 = utils.rot3d(2, 2. * self.randg.rand(1) * np.pi)
            trans = T1 @ M2 @ np.linalg.inv(T0)
            xyz0 = utils.apply_transform(xyz0, T0)
            xyz1 = utils.apply_transform(xyz1, T1)
        else:
            trans = M2

        ids0, points0, ids_xy0, points_xy0 = utils.load_voxel(
            xyz0, coords_range_xyz=self.coords_range_xyz, div_n=self.div)
        ids1, points1, ids_xy1, points_xy1 = utils.load_voxel(
            xyz1, coords_range_xyz=self.coords_range_xyz, div_n=self.div)

        voxel_out0 = np.zeros(self.div, dtype='float32')
        voxel_out0[ids0[:, 0], ids0[:, 1], ids0[:, 2]] = 1
        voxel_out1 = np.zeros(self.div, dtype='float32')
        voxel_out1[ids1[:, 0], ids1[:, 1], ids1[:, 2]] = 1
        time1 = time.time()
        return {
            "points0": points0,
            "points_xy0": points_xy0,
            "id0": ids0,
            "ids_xy0": ids_xy0,
            "voxel0": voxel_out0,
            "points1": points1,
            "points_xy1": points_xy1,
            "id1": ids1,
            "ids_xy1": ids_xy1,
            "voxel1": voxel_out1,
            "trans0": trans.astype('float32'),
            "trans1": np.identity(4, dtype='float32'),
            "time": time1 - time0
        }


if __name__ == '__main__':
    # 测试Chilean数据集
    dataset = ChileanDatasetOverlap(
        sessions=['100', '101'],
        root="/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/chilean_NoRot_NoScale/",
        pos_threshold_min=0,
        pos_threshold_max=7,
        neg_thresgold=35,
        coords_range_xyz=[-10., -10, -4, 10, 10, 8],
        div_n=[256, 256, 32],
        random_rotation=True,
        random_occ=True)
    for i in range(len(dataset)):
        d = dataset[random.randint(0, len(dataset) - 1)]
        print(d['voxel0'].shape, d['trans0'])
        break