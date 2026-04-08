
from math import radians, sin, cos, sqrt, atan2
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from utils.timefeatures import time_features
import warnings
import torch

warnings.filterwarnings('ignore')


def clean_feature_name(name):
    return str(name).replace('\n', ' ').strip()


class Dataset_wind_data_graph(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='M', data_path='wind_data.csv',
                 target='station1', scale=True, timeenc=0, freq='h', data_step=1,
                 min_num_nodes=2, c_out=2, **_):
        self.seq_len, self.label_len, self.pred_len = size
        self.data_step = data_step
        self.total_seq_len = self.seq_len + self.pred_len
        self.flag, self.features, self.scale, self.timeenc, self.freq = flag, features, scale, timeenc, freq
        self.root_path, self.data_path = root_path, data_path
        self.window_size = 3
        self.c_out = c_out
        self.__read_data__()

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(radians, [lat1, lon1, lat2, lon2])
        dlon, dlat = lon2_rad - lon1_rad, lat2_rad - lat1_rad
        a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def __read_data__(self):
        self.scaler = MinMaxScaler()
        _path = os.path.join(self.root_path, self.flag, self.data_path)
        df_raw = pd.read_csv(_path, header=[0, 1])

        df_raw.columns = pd.MultiIndex.from_tuples(
            [(clean_feature_name(name), station) for name, station in df_raw.columns]
        )

        features_to_use = [
            'Wind speed Sensor 1 (m/s)', 'Wind direction, Minimum (deg)', 'Grid current (A)',
            'Power, Maximum (kW)', 'Capacity factor', 'Energy Export (kWh)', 'Power (kW)'
        ]

        if self.c_out == 1:
            target_features = ['Power (kW)']
        else:
            target_features = ['Wind speed Sensor 1 (m/s)', 'Power (kW)']

        other_features = [f for f in features_to_use if f not in target_features]
        final_features_order = other_features + target_features

        self.features_list = final_features_order
        self.target_feature_indices = [self.features_list.index(f) for f in target_features]
        self.power_feature_index = self.features_list.index('Power (kW)')
        self.power_target_index = target_features.index('Power (kW)')

        available_features = df_raw.columns.get_level_values(0).unique()
        for f in self.features_list:
            if f not in available_features:
                raise ValueError(
                    f"Fatal: The required feature '{f}' was not found. Available: {available_features.tolist()}")

        time_col_tuple = ('time', 'station1')
        all_stations = sorted(df_raw.columns.get_level_values(1).unique().tolist())
        # Filter out 'time' columns from stations list if present
        if 'time' in all_stations: all_stations.remove('time')

        cols_to_keep = [time_col_tuple] + [(feat, stat) for feat in self.features_list for stat in all_stations if
                                           (feat, stat) in df_raw.columns]

        df_filtered = df_raw[cols_to_keep]
        timestamps = pd.to_datetime(df_filtered[time_col_tuple])
        df_data = df_filtered.drop(columns='time', level=0)
        df_data.set_index(timestamps, inplace=True)

        stations = sorted(df_data.columns.get_level_values(1).unique().tolist())
        self._stations = {s: i for i, s in enumerate(stations)}
        self._stations_inv = {v: k for k, v in self._stations.items()}

        df_data = df_data.reindex(columns=stations, level=1)
        df_data = df_data.reindex(columns=self.features_list, level=0)
        df_data = df_data.fillna(method='ffill').fillna(method='bfill')

        station_info_path = os.path.join(self.root_path, 'station_info.csv')
        self.station_info = pd.read_csv(station_info_path)
        self.station_info['name'] = self.station_info['name'].apply(lambda x: x.replace(' ', ''))

        if self.scale:
            train_path = os.path.join(self.root_path, 'train', self.data_path)
            if self.flag != 'train' and os.path.exists(train_path):
                train_df_raw = pd.read_csv(train_path, header=[0, 1])
                train_df_raw.columns = pd.MultiIndex.from_tuples(
                    [(clean_feature_name(name), station) for name, station in train_df_raw.columns]
                )

                # Use df_data's columns to filter train_df_raw to ensure consistency
                train_df = train_df_raw[df_data.columns]
                train_df = train_df.fillna(method='ffill').fillna(method='bfill')
                self.scaler.fit(train_df.values)
            else:
                self.scaler.fit(df_data.values)
            self.data_x = self.scaler.transform(df_data.values)
        else:
            self.data_x = df_data.values

        self.data_x = self.data_x.reshape(len(df_data), len(stations), len(self.features_list))
        self._build_graphs(df_data)

        df_stamp = df_data.index.to_frame(index=False, name='time')
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.time.dt.month;
            df_stamp['day'] = df_stamp.time.dt.day
            df_stamp['weekday'] = df_stamp.time.dt.weekday;
            df_stamp['hour'] = df_stamp.time.dt.hour
            df_stamp['minute'] = df_stamp.time.dt.minute // 10
            self.data_stamp = df_stamp.drop(['time'], axis=1).values
        else:
            self.data_stamp = time_features(pd.to_datetime(df_stamp['time'].values), freq=self.freq).transpose(1, 0)

        nan_mask = np.isnan(self.data_x).any(axis=(1, 2))
        valid_start = 0;
        valid_indices_list = []
        for i, is_nan in enumerate(nan_mask):
            if is_nan:
                if i - valid_start >= self.total_seq_len:
                    valid_indices_list.append(np.arange(valid_start, i - self.total_seq_len + 1))
                valid_start = i + 1
        if len(df_data) - valid_start >= self.total_seq_len:
            valid_indices_list.append(np.arange(valid_start, len(df_data) - self.total_seq_len + 1))

        self.valid_indxs = np.concatenate(valid_indices_list)[::self.data_step] if valid_indices_list else np.array([])
        self.df_wind_raw = df_data
    def _build_graphs(self, df_data):
            stations = list(self._stations.keys())
            num_stations = len(stations)
            distance_matrix = np.zeros((num_stations, num_stations))
            
            # 1. 计算静态距离矩阵
            for i, stat_i in enumerate(stations):
                for j, stat_j in enumerate(stations):
                    info_i = self.station_info[self.station_info['name'] == stat_i].iloc[0]
                    info_j = self.station_info[self.station_info['name'] == stat_j].iloc[0]
                    distance_matrix[i, j] = self.haversine(info_i.lat, info_i.lon, info_j.lat, info_j.lon)
            
            # 2. 准备数据
            wind_speed_data = df_data.xs('Wind speed Sensor 1 (m/s)', level=0, axis=1).values
            power_data = df_data.xs('Power (kW)', level=0, axis=1).values
            wind_dir_data = df_data.xs('Wind direction, Minimum (deg)', level=0, axis=1).values
            
            # --- 静态图构建 (保持不变) ---
            senders_s, receivers_s, edges_s = [], [], []
            for i in range(num_stations):
                for j in range(num_stations):
                    senders_s.append(i)
                    receivers_s.append(j)
                    # 静态图边特征占位，保持维度一致性
                    edges_s.append([1.0, distance_matrix[i, j], 0.0, 0.0, 0.0, 0.0, 0.0]) 
            self.static_graph = {'edges': np.array(edges_s, dtype=np.float32), 
                                'senders': np.array(senders_s),
                                'receivers': np.array(receivers_s)}

            # --- 动态图构建 (核心修改) ---
            self.dynamic_graphs = []
            dir_threshold, distance_threshold = 30.0, 50.0 # 相似度计算的衰减参数
            
            # 【修改点1】定义连通性策略：
            # 由于我们要在模型里学权重，这里不能用动态权重过滤边。
            # 策略：使用 K-近邻 (KNN) 或 距离阈值 来确定“潜在的边”。
            # 这里为了简单有效，我们保留距离小于某阈值的边，或者所有边(全连接)。
            # 建议：对于小规模风场(如Kelmarsh)，使用全连接图，让模型自己学权重。
            
            for t in range(len(df_data)):
                senders_d, receivers_d, edges_d = [], [], []
                
                # 动态阈值参数
                speed_std = np.std(wind_speed_data[t])
                power_std = np.std(power_data[t])
                speed_scale = max(speed_std * 1.5, 2.0)
                power_scale = max(power_std * 1.5, 0.5)
                
                for i in range(num_stations):
                    for j in range(num_stations):
                        # 1. 计算原始差异
                        if t >= self.window_size:
                            start_idx = t - self.window_size
                            dir_diff = np.mean(np.abs(wind_dir_data[start_idx:t, i] - wind_dir_data[start_idx:t, j]))
                            speed_diff = np.mean(np.abs(wind_speed_data[start_idx:t, i] - wind_speed_data[start_idx:t, j]))
                            power_diff = np.mean(np.abs(power_data[start_idx:t, i] - power_data[start_idx:t, j]))
                        else:
                            dir_diff = abs(wind_dir_data[t, i] - wind_dir_data[t, j])
                            speed_diff = abs(wind_speed_data[t, i] - wind_speed_data[t, j])
                            power_diff = abs(power_data[t, i] - power_data[t, j])
                        
                        distance = distance_matrix[i, j]

                        # 2. 计算四个归一化的相似度分量 (0~1之间)
                        # S_speed, S_dir, S_power, S_dist
                        sim_speed = np.exp(-speed_diff / speed_scale)
                        sim_dir = np.exp(-dir_diff / dir_threshold)
                        sim_power = np.exp(-power_diff / power_scale)
                        sim_dist = np.exp(-distance / distance_threshold)

                        # 【修改点2】不再计算最终 edge_weight，也不进行阈值过滤
                        # 只要不是太远(可选)，都保留，把四个分量存入特征
                        # 如果为了节省显存，可以加一个 if distance < 5000: ...
                        
                        senders_d.append(i)
                        receivers_d.append(j)
                        
                        # 【修改点3】构造新的边特征向量
                        # 前4位是相似度分量（用于模型学习权重），后面是原始物理量
                        # Feature: [Sim_Speed, Sim_Dir, Sim_Power, Sim_Dist, Raw_Dist, Raw_Speed_Diff, Raw_Dir_Diff]
                        edge_feature = [sim_speed, sim_dir, sim_power, sim_dist, distance, speed_diff, dir_diff]
                        edges_d.append(edge_feature)

                self.dynamic_graphs.append({
                    'edges': np.array(edges_d, dtype=np.float32), 
                    'senders': np.array(senders_d),
                    'receivers': np.array(receivers_d)
                })

    def __getitem__(self, index):
        s_begin = self.valid_indxs[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        x_mark = self.data_stamp[s_begin:s_end]

        dec_inp_known = self.data_x[r_begin:s_end]
        # FIX in placeholder creation
        dec_inp_placeholder = np.zeros((self.pred_len, dec_inp_known.shape[1], dec_inp_known.shape[2]))
        dec_inp = np.concatenate([dec_inp_known, dec_inp_placeholder], axis=0)

        y_mark = self.data_stamp[r_begin:r_end]
        seq_y = self.data_x[s_end:r_end]

        dyn_graph_struct = self.dynamic_graphs[s_begin]

        x_dyn = {'nodes': seq_x.transpose(1, 0, 2), 'station_names': list(self._stations.keys()), **dyn_graph_struct}
        x_stat = {'nodes': seq_x.transpose(1, 0, 2), 'station_names': list(self._stations.keys()), **self.static_graph}

        dec_inp_graph = {'nodes': dec_inp.transpose(1, 0, 2), 'station_names': list(self._stations.keys()),
                         **dyn_graph_struct}
        y_target = {'nodes': seq_y.transpose(1, 0, 2), 'station_names': list(self._stations.keys())}

        return [x_dyn, x_stat], dec_inp_graph, y_target, x_mark, y_mark

    def __len__(self):
        return len(self.valid_indxs)
    def inverse_transform(self, data):
            # data shape: (total_nodes, pred_len, c_out)
            # total_nodes = batch_size * num_stations
            
            # 确保输入是 numpy 数组
            if isinstance(data, torch.Tensor):
                data = data.cpu().detach().numpy()
                
            num_nodes, pred_len, c_out = data.shape
            num_stations = len(self._stations)
            # num_features = len(self.features_list) # 暂时没用到，注释掉防止警告
            
            # 1. 计算 Batch Size
            batch_size = num_nodes // num_stations
            
            # 2. 重塑数据以便操作
            # 原始数据混合了 Batch 和 Station，我们需要把它们分开
            # 假设输入顺序是 [Batch1_S1, Batch1_S2 ... Batch1_S6, Batch2_S1 ...]
            data = data.reshape(batch_size, num_stations, pred_len, c_out)
            
            # 3. 调整维度顺序 -> [Batch, PredLen, Station, Feature] -> Flatten
            # 我们需要把时间展平，保留 Station 和 Feature 维度
            # === FIX: 使用 transpose 替代 permute ===
            data_permuted = data.transpose(0, 2, 1, 3) # [Batch, PredLen, Station, Feature]
            data_flat = data_permuted.reshape(-1, num_stations, c_out) # [Total_Time_Steps, Station, Feature]
            
            total_time_steps = data_flat.shape[0]
            
            # 4. 创建反归一化占位符
            # Scaler 的列数 = num_features * num_stations
            placeholder = np.zeros((total_time_steps, self.scaler.n_features_in_))
            
            # 5. 【核心修正】正确填充占位符
            # 数据的列顺序是：[Feature1_S1, Feature1_S2 ... Feature1_S6, Feature2_S1 ...]
            
            for i in range(c_out):
                # 获取当前输出特征在原始特征列表中的索引
                feature_idx_in_full = self.target_feature_indices[i]
                
                for stat_idx in range(num_stations):
                    # === 修正公式 ===
                    # 正确的索引 = 特征索引 * 站点总数 + 当前站点索引
                    col_idx = feature_idx_in_full * num_stations + stat_idx
                    
                    # 填入数据
                    placeholder[:, col_idx] = data_flat[:, stat_idx, i]
                    
            # 6. 执行反归一化
            inv_placeholder = self.scaler.inverse_transform(placeholder)
            
            # 7. 提取还原后的数据
            inv_data_flat = np.zeros_like(data_flat)
            
            for i in range(c_out):
                feature_idx_in_full = self.target_feature_indices[i]
                for stat_idx in range(num_stations):
                    # 使用相同的修正公式取回数据
                    col_idx = feature_idx_in_full * num_stations + stat_idx
                    inv_data_flat[:, stat_idx, i] = inv_placeholder[:, col_idx]
                    
            # 8. 恢复原始形状 (Total_Nodes, Pred_Len, C_Out)
            # 当前: [Total_Time, Station, Feature]
            # 目标: [Batch * Station, PredLen, Feature]
            
            inv_data_reshaped = inv_data_flat.reshape(batch_size, pred_len, num_stations, c_out)
            # === FIX: 使用 transpose 替代 permute ===
            # 变回 [Batch, Station, PredLen, Feature]
            inv_data_final = inv_data_reshaped.transpose(0, 2, 1, 3).reshape(num_nodes, pred_len, c_out)
            
            return inv_data_final

class Dataset_ERA5_Wind_Graph(Dataset_wind_data_graph):
    def __init__(self, root_path, flag='train', size=None, features='MS', data_path='wind_data.csv',
                 target='station1', scale=True, timeenc=0, freq='h', data_step=1,
                 min_num_nodes=2, era5_data_path='combined_data.csv',
                 era5_center_lat=52.25, era5_center_lon=-1.0, enc_in=7, c_out=2, **_):
        self.era5_data_path = era5_data_path
        self.era5_center_coords = (era5_center_lat, era5_center_lon)
        self.enc_in = enc_in
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq,
                         data_step=data_step, min_num_nodes=min_num_nodes, c_out=c_out)

    def __read_data__(self):
        super().__read_data__()
        self.__read_era5_data__()
        common_timestamps = self.df_wind_raw.index.intersection(self.df_era5_pivot.index)
        wind_indices = self.df_wind_raw.index.isin(common_timestamps)
        self.data_x = self.data_x[wind_indices]
        self.data_stamp = self.data_stamp[wind_indices]
        self.dynamic_graphs = [self.dynamic_graphs[i] for i, keep in enumerate(wind_indices) if keep]
        self.df_wind_raw = self.df_wind_raw.loc[common_timestamps]
        self.era5_data_x = self.df_era5_pivot.loc[common_timestamps].values.reshape(len(common_timestamps),
                                                                                    len(self.era5_stations), -1)
        nan_mask = np.isnan(self.data_x).any(axis=(1, 2))
        valid_start = 0;
        valid_indices_list = []
        for i, is_nan in enumerate(nan_mask):
            if is_nan:
                if i - valid_start >= self.total_seq_len:
                    valid_indices_list.append(np.arange(valid_start, i - self.total_seq_len + 1))
                valid_start = i + 1
        if len(common_timestamps) - valid_start >= self.total_seq_len:
            valid_indices_list.append(np.arange(valid_start, len(common_timestamps) - self.total_seq_len + 1))
        self.valid_indxs = np.concatenate(valid_indices_list)[::self.data_step] if valid_indices_list else np.array([])

    def __read_era5_data__(self):
        df_era5 = pd.read_csv(self.era5_data_path)
        df_era5['valid_time'] = pd.to_datetime(df_era5['valid_time'])
        df_era5['coords'] = list(zip(df_era5['latitude'], df_era5['longitude']))
        unique_coords = df_era5['coords'].unique()
        self.era5_stations = {coords: i for i, coords in enumerate(unique_coords)}
        self.era5_stations_inv = {i: coords for coords, i in self.era5_stations.items()}
        center_tuple = self.era5_center_coords
        if center_tuple not in self.era5_stations:
            distances = {coords: self.haversine(center_tuple[0], center_tuple[1], coords[0], coords[1]) for coords in
                         self.era5_stations.keys()}
            center_tuple = min(distances, key=distances.get)
        self.era5_center_node_idx = self.era5_stations[center_tuple]
        era5_features_to_use = ['t2m', 'blh', 'ws10', 'wd10', 'ws100', 'wd100']
        self.df_era5_pivot = df_era5.pivot_table(index='valid_time', columns='coords', values=era5_features_to_use)
        self.df_era5_pivot.columns = self.df_era5_pivot.columns.swaplevel(0, 1)
        self.df_era5_pivot.sort_index(axis=1, level=0, inplace=True)
        self.df_era5_pivot.fillna(method='ffill', inplace=True)
        self.df_era5_pivot.fillna(method='bfill', inplace=True)
        num_nodes = len(self.era5_stations)
        senders, receivers, edge_feats = [], [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                senders.append(i);
                receivers.append(j)
                lat1, lon1 = self.era5_stations_inv[i];
                lat2, lon2 = self.era5_stations_inv[j]
                edge_feats.append([self.haversine(lat1, lon1, lat2, lon2), lat1 - lat2, lon1 - lon2])
        self.era5_static_graph = {'edges': np.array(edge_feats, dtype=np.float32), 'senders': np.array(senders),
                                  'receivers': np.array(receivers)}

    def __getitem__(self, index):
        s_begin = self.valid_indxs[index]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x_wind = self.data_x[s_begin:s_end]
        seq_x_era5 = self.era5_data_x[s_begin:s_end]
        x_mark = self.data_stamp[s_begin:s_end]

        dec_inp_known = self.data_x[r_begin:s_end]
        dec_inp_placeholder = np.zeros((self.pred_len, dec_inp_known.shape[1], dec_inp_known.shape[2]))
        dec_inp_wind = np.concatenate([dec_inp_known, dec_inp_placeholder], axis=0)

        y_mark = self.data_stamp[r_begin:r_end]
        seq_y_wind = self.data_x[s_end:r_end]

        dyn_graph_struct = self.dynamic_graphs[s_begin]

        graph_x_wind_dyn = {'nodes': seq_x_wind.transpose(1, 0, 2), 'station_names': list(self._stations.keys()),
                            **dyn_graph_struct}
        graph_x_wind_stat = {'nodes': seq_x_wind.transpose(1, 0, 2), 'station_names': list(self._stations.keys()),
                             **self.static_graph}
        graph_x_era5 = {'nodes': seq_x_era5.transpose(1, 0, 2), 'station_names': list(self.era5_stations.keys()),
                        **self.era5_static_graph}
        batch_x = {'wind_dynamic': graph_x_wind_dyn, 'wind_static': graph_x_wind_stat, 'era5': graph_x_era5}

        batch_dec_inp = {'nodes': dec_inp_wind.transpose(1, 0, 2), 'station_names': list(self._stations.keys()),
                         **dyn_graph_struct}
        graph_y_target = {'nodes': seq_y_wind.transpose(1, 0, 2), 'station_names': list(self._stations.keys())}

        return batch_x, batch_dec_inp, graph_y_target, x_mark, y_mark, self.era5_center_node_idx


def collate_graph(batch):
    is_era5 = isinstance(batch[0][0], dict)

    def batch_single_stream_graphs(graph_list):
        if not graph_list or all(g is None for g in graph_list): return None
        nodes_list = [g['nodes'] for g in graph_list if g is not None]
        if not nodes_list: return None
        nodes = np.concatenate(nodes_list, axis=0)
        edges_list = [g['edges'] for g in graph_list if
                      g is not None and g.get('edges') is not None and g['edges'].size > 0]
        if edges_list:
            edges = np.concatenate(edges_list, axis=0)
        else:
            num_edge_feats = 5
            for g in graph_list:
                if g is not None and g.get('edges') is not None and g['edges'].ndim == 2:
                    num_edge_feats = g['edges'].shape[1]
                    break
            edges = np.empty((0, num_edge_feats), dtype=np.float32)
        n_node = np.array([g['nodes'].shape[0] for g in graph_list if g is not None])
        n_edge = np.array(
            [g['edges'].shape[0] if g is not None and g.get('edges') is not None else 0 for g in graph_list])
        senders, receivers, node_offset = [], [], 0
        graph_indices = np.where(n_node > 0)[0]
        for i in graph_indices:
            g = graph_list[i]
            if g.get('senders') is not None and len(g['senders']) > 0:
                senders.append(g['senders'] + node_offset)
                receivers.append(g['receivers'] + node_offset)
            node_offset += n_node[i]
        senders = np.concatenate(senders) if senders else np.array([], dtype=int)
        receivers = np.concatenate(receivers) if receivers else np.array([], dtype=int)
        station_names = np.concatenate([g['station_names'] for g in graph_list if g is not None])
        return {'nodes': nodes, 'edges': edges, 'senders': senders, 'receivers': receivers, 'n_node': n_node,
                'n_edge': n_edge, 'station_names': station_names}

    if is_era5:
        batch_x = {
            'wind_dynamic': batch_single_stream_graphs([s[0]['wind_dynamic'] for s in batch]),
            'wind_static': batch_single_stream_graphs([s[0]['wind_static'] for s in batch]),
            'era5': batch_single_stream_graphs([s[0]['era5'] for s in batch]),
        }
        batch_dec_inp = batch_single_stream_graphs([s[1] for s in batch])
        batch_y_target = batch_single_stream_graphs([s[2] for s in batch])
        seq_x_mark = torch.from_numpy(np.stack([s[3] for s in batch]))
        seq_y_mark = torch.from_numpy(np.stack([s[4] for s in batch]))
        center_indices = torch.from_numpy(np.array([s[5] for s in batch]))
        return batch_x, batch_dec_inp, batch_y_target, seq_x_mark, seq_y_mark, center_indices
    else:  # WindGraph
        x_dyn = batch_single_stream_graphs([s[0][0] for s in batch])
        x_stat = batch_single_stream_graphs([s[0][1] for s in batch])
        batch_x = [x_dyn, x_stat]
        batch_dec_inp = batch_single_stream_graphs([s[1] for s in batch])
        batch_y_target = batch_single_stream_graphs([s[2] for s in batch])
        x_mark = torch.from_numpy(np.stack([s[3] for s in batch]))
        y_mark = torch.from_numpy(np.stack([s[4] for s in batch]))
        return batch_x, batch_dec_inp, batch_y_target, x_mark, y_mark