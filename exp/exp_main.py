import os
import torch
import numpy as np
import torch.nn as nn
from torch import optim
import time
import warnings
import torch.nn.functional as F

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import GraphTransformer
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.graph_utils import data_dicts_to_graphs_tuple
from utils.CustomDataParallel import DataParallelGraph
from utils.physics import PhysicalPowerCurve 

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):

    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        
        self.initial_physics_weight = getattr(args, 'physics_weight', 0.1)
        self.current_physics_weight = self.initial_physics_weight
        self.physics_decay_rate = getattr(args, 'physics_decay_rate', 1.0)

        temp_data, _ = self._get_data(flag='train')
        self.target_indices = temp_data.target_feature_indices
        self.power_target_index = temp_data.power_target_index

        if args.c_out >= 2:
            ws_idx_in_features = temp_data.features_list.index('Wind speed Sensor 1 (m/s)')
            self.v_max_val = temp_data.scaler.data_max_[ws_idx_in_features]
            
            self.v_rated_val = args.v_rated
            
            self.kappa = self.v_max_val / self.v_rated_val
            print(f"Physics Params Initialized: v_max={self.v_max_val:.2f}, v_rated={self.v_rated_val:.2f}, kappa={self.kappa:.4f}")
        else:
            self.kappa = 1.0 
            self.v_max_val = 25.0 # 默认防报错

        del temp_data
    def _build_model(self):
        model = GraphTransformer.Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = DataParallelGraph(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate,weight_decay=1e-4 )

    def _select_criterion(self):
        return nn.MSELoss()
    def physics_loss(self, pred):

        if pred.shape[-1] < 2: 
            return torch.tensor(0.0, device=pred.device)
        

        w_norm = pred[..., 0]  
        p_norm = pred[..., 1]  
 
        w_norm = F.relu(w_norm)
        p_norm = F.relu(p_norm)
        theoretical_power = torch.clamp((self.kappa * w_norm).pow(3), max=1.0)
       
        v_in_threshold = 3.0 / self.v_max_val
        v_rated_threshold = 1.0 / self.kappa  # 或者用 0.95 近似
        
        mask_ramp = (w_norm > v_in_threshold) & (w_norm < v_rated_threshold)
        
        mask_non_sat = p_norm < 0.98
        
        valid_mask = mask_ramp & mask_non_sat
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device)

        loss_element_wise = F.mse_loss(p_norm, theoretical_power, reduction='none')
        
        masked_loss = loss_element_wise * valid_mask.float()
        
        loss = masked_loss.sum() / (valid_mask.float().sum() + 1e-8)

        return loss
    def _process_one_batch(self, batch_data, for_test=False):
        if self.args.data == 'ERA5WindGraph':
            x_enc_dict, dec_inp_dict, y_target_dict, x_mark, y_mark, center_idx = batch_data
            x_enc = {k: data_dicts_to_graphs_tuple(v, self.device) for k, v in x_enc_dict.items()}
            dec_inp = data_dicts_to_graphs_tuple(dec_inp_dict, self.device)
            station_ids = x_enc['wind_dynamic'].station_names if for_test else None
        else:  # WindGraph
            x_enc_list, dec_inp_dict, y_target_dict, x_mark, y_mark = batch_data
            x_enc = [data_dicts_to_graphs_tuple(g, self.device) for g in x_enc_list]
            dec_inp = data_dicts_to_graphs_tuple(dec_inp_dict, self.device)
            center_idx = None
            station_ids = x_enc[0].station_names if for_test else None

        y_target = data_dicts_to_graphs_tuple(y_target_dict, self.device)
        x_mark = x_mark.float().to(self.device)
        y_mark = y_mark.float().to(self.device)

        return x_enc, dec_inp, y_target, x_mark, y_mark, center_idx, station_ids

    def _run_model_and_get_loss(self, batch_data, criterion):
        x_enc, dec_inp, y_target, x_mark, y_mark, center_idx, _ = self._process_one_batch(batch_data)

        outputs = self.model(x_enc, x_mark, dec_inp, y_mark, center_indices=center_idx)

        outputs_nodes = outputs.nodes
        pred = outputs_nodes[:, -self.args.pred_len:, :]

        true_all_features = y_target.nodes
        true = true_all_features[:, :, self.target_indices]

        if self.args.c_out > 1:
            loss_ws = criterion(pred[..., 0], true[..., 0])
            loss_p = criterion(pred[..., 1], true[..., 1])
            data_loss = 0.3 * loss_ws + 0.7 * loss_p
        else:
            data_loss = criterion(pred, true)

        phys_loss = self.physics_loss(pred)
        
        total_loss = data_loss + self.current_physics_weight * phys_loss
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        path = os.path.join(self.args.checkpoints, setting);
        os.makedirs(path, exist_ok=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, checkpoint=self.args.checkpoint_flag,
                                       model_setup=self.args)

        for epoch in range(self.args.train_epochs):
            self.current_physics_weight = self.initial_physics_weight * (self.physics_decay_rate ** epoch)
            print(f"Epoch: {epoch + 1} | Current Physics Weight: {self.current_physics_weight:.6f}")
            self.model.train()
            train_loss = []
            for i, batch_data in enumerate(train_loader):

                model_optim.zero_grad()
                loss = self._run_model_and_get_loss(batch_data, criterion)
                train_loss.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                model_optim.step()

            train_loss_avg = np.average(train_loss)
            vali_loss_avg = self.vali(vali_loader, criterion)
            print(f"Epoch: {epoch + 1} | Train Loss: {train_loss_avg:.7f} | Vali Loss: {vali_loss_avg:.7f}")
            early_stopping(vali_loss_avg, self.model, path, epoch)
            if early_stopping.early_stop: break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, batch_data in enumerate(vali_loader):
                loss = self._run_model_and_get_loss(batch_data, criterion)
                total_loss.append(loss.item())
        self.model.train()
        return np.average(total_loss)
    def test(self, setting, test=1, base_dir='', save_dir=None):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            load_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))

        all_pred_targets_full = []
        all_true_targets_full = []
        all_station_ids = []

        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                x_enc, dec_inp, y_target, x_mark, y_mark, center_idx, station_ids_batch = self._process_one_batch(
                    batch_data, for_test=True)

                outputs = self.model(x_enc, x_mark, dec_inp, y_mark, center_indices=center_idx)

                pred_raw = outputs.nodes[:, -self.args.pred_len:, :]

                pred_sliced = pred_raw.detach().cpu().numpy()
                
                true_sliced = y_target.nodes[:, :, self.target_indices].detach().cpu().numpy()
                all_pred_targets_full.append(pred_sliced)
                all_true_targets_full.append(true_sliced)
                all_station_ids.append(station_ids_batch)

        pred_targets_all = np.concatenate(all_pred_targets_full, axis=0)
        true_targets_all = np.concatenate(all_true_targets_full, axis=0)
        station_ids = np.concatenate(all_station_ids, axis=0)
        preds_power_scaled = pred_targets_all[..., self.power_target_index].reshape(-1, 1)
        trues_power_scaled = true_targets_all[..., self.power_target_index].reshape(-1, 1)

        mae, mse, rmse, mape, mspe, r2 = metric(preds_power_scaled, trues_power_scaled)
        print(f'Overall Scaled Power -> MSE:{mse:.4f}, MAE:{mae:.4f}, R2:{r2:.4f}')

        # 反归一化
        preds_un = test_data.inverse_transform(pred_targets_all)
        trues_un = test_data.inverse_transform(true_targets_all)

        preds_un_power = preds_un[..., self.power_target_index].reshape(-1, 1)
        trues_un_power = trues_un[..., self.power_target_index].reshape(-1, 1)

        mae_un, mse_un, rmse_un, mape_un, mspe_un, r2_un = metric(preds_un_power, trues_un_power)
        print(f'Overall Unscaled Power -> MSE:{mse_un:.4f}, MAE:{mae_un:.4f}, R2:{r2_un:.4f}')

        # 保存结果
        folder_path = os.path.join(base_dir or './', 'results', setting)
        os.makedirs(folder_path, exist_ok=True)

        np.save(os.path.join(folder_path, 'pred.npy'), preds_un_power.reshape(pred_targets_all.shape[:-1] + (1,)))
        np.save(os.path.join(folder_path, 'true.npy'), trues_un_power.reshape(true_targets_all.shape[:-1] + (1,)))
        np.save(os.path.join(folder_path, 'station_ids.npy'), station_ids)

        unique_stations = np.unique(station_ids)
        preds_un_power_shaped = preds_un_power.reshape(station_ids.shape[0], -1)
        trues_un_power_shaped = trues_un_power.reshape(station_ids.shape[0], -1)

        for stat_name in unique_stations:
            indices = np.where(station_ids == stat_name)[0]
            if len(indices) > 0:
                mae_i, mse_i, rmse_i, mape_i, mspe_i, r2_i = metric(preds_un_power_shaped[indices].flatten(),
                                                                    trues_un_power_shaped[indices].flatten())
                print(f"Metrics for {stat_name} (Unscaled Power): MAE={mae_i:.4f}, MSE={mse_i:.4f}, R2={r2_i:.4f}")